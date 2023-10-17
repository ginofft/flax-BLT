import numpy as np
import flax
import jax
from jax import lax
import jax.numpy as jnp
import functools
from src import sampling
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display

@flax.struct.dataclass
class State:
    cur_index: jax.Array
    cur_seqs: jax.Array
    rng: jax.Array
    final_seqs: jax.Array

def state_init(masked_batch, rng, total_iteration_num):
    cur_index0 = jnp.array(0)
    cur_seqs0 = masked_batch
    final_seqs0 = jnp.expand_dims(masked_batch, 1)
    final_seqs0 = jnp.tile(final_seqs0, (1, total_iteration_num, 1))
    return State(
        cur_index=cur_index0,
        cur_seqs=cur_seqs0,
        rng=rng,
        final_seqs=final_seqs0
    )

class LayoutDecoder:
    def __init__(self, 
                 vocab_size, 
                 seq_len, 
                 layout_dim,
                 id_to_label, 
                 color_map,
                 resolution_w = 32, 
                 resolution_h = 32,
                 iterative_nums = np.array([3,3,3]), 
                 temperature = 1.0):
        
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.layout_dim = layout_dim
        self.iterative_nums = iterative_nums
        self.temperature = temperature
        self.id_to_label = id_to_label
        self.color_map = color_map
        self.resolution_w = resolution_w
        self.resolution_h = resolution_h

    def decode(self, 
               model, 
               inputs, 
               logit_masks, 
               sampling_method='topp', 
               rng=jax.random.PRNGKey(236)):
        
        if model is not None:
            self.model = model
        if self.model is None:
            raise TypeError("No model found!!")
        
        layout_dim = self.layout_dim
        total_dim = self.layout_dim*2 + 1
        cum_iterative_num = np.cumsum(self.iterative_nums)
        total_iteration = np.sum(self.iterative_nums)
        init_state = state_init(inputs, rng, total_iteration)

        position_ids = jnp.arange(inputs.shape[-1])[None, :]
        is_asset = position_ids % total_dim == 0
        is_size = functools.reduce(
            lambda x,y: x|y,
            [position_ids % total_dim == i for i in range(1, layout_dim+1)])
        is_position = functools.reduce(
            lambda x,y: x|y,
            [position_ids % total_dim == i for i in range(layout_dim+1, total_dim)])
        special_symbol_mask = jnp.ones((1, logit_masks.shape[1], 4))
        logit_masks = jax.lax.dynamic_update_slice(logit_masks, special_symbol_mask, (0,0,0))
        
        def loop_cond_fn(state):
            return (state.cur_index < total_iteration)
        
        def loop_body_fn(state):
            cur_ids = state.cur_seqs
            
            logits = self.tokens_to_logits(cur_ids)
            logits = jax.nn.log_softmax(logits, axis = -1)
            logits = jnp.where(logit_masks>0, -1e7, logits)
            rng = state.rng
            step = state.cur_index
            if sampling_method == 'greedy':
                sampled_ids = jnp.argmax(logits, axis=-1)
            else: 
                rng, sample_rng = jax.random.split(rng, 2)
                sampled_ids = jnp.argmax(logits, axis=-1)
                sample_ids_2nd = sampling.sampling(
                    logits, sample_rng, topk=5, temperature=self.temperature
                )
                cur_attribute = jnp.array(step >= cum_iterative_num[1], dtype='int32')+jnp.array(
                step >= cum_iterative_num[0], dtype='int32')
                sampled_ids = jnp.where(cur_attribute>1, sampled_ids, sample_ids_2nd)
            
            def position_iteration_mask(sampled_ids):
                sampled_ids = jnp.where(cur_ids==3, sampled_ids, cur_ids)
                masked_ratio = (cum_iterative_num[2] - step - 1.) / self.iterative_nums[2]
                target_mask = (cur_ids==3) & is_position
                return sampled_ids, masked_ratio, target_mask
            
            def size_iteration_mask(sampled_ids):
                sampled_ids = jnp.where((cur_ids==3) & (~is_position), sampled_ids,
                                        cur_ids)
                masked_ratio = (cum_iterative_num[1] - step - 1.) / self.iterative_nums[1]
                target_mask = (cur_ids==3) & is_size
                return sampled_ids, masked_ratio, target_mask
            
            def asset_iteration_mask(sampled_ids):
                sampled_ids = jnp.where((cur_ids == 3) & (is_asset), sampled_ids, cur_ids)
                masked_ratio = (cum_iterative_num[0] - step - 1.) / self.iterative_nums[0]
                target_mask = (cur_ids == 3) & is_asset
                return sampled_ids, masked_ratio, target_mask
            
            cur_attribute = jnp.array(
                step >= cum_iterative_num[1], dtype='int32') + jnp.array(
                    step >= cum_iterative_num[0], dtype='int32')
            cur_mask_info = jax.lax.switch(
                cur_attribute,
                [asset_iteration_mask, size_iteration_mask, position_iteration_mask],
                sampled_ids)

            sampled_ids, masked_ratio, target_mask = cur_mask_info
            final_seqs = jax.lax.dynamic_update_slice(
                state.final_seqs, jnp.expand_dims(sampled_ids, axis=1), (0, step, 0))
            
            probs = jax.nn.softmax(logits, axis=-1)
            selected_probs = jnp.squeeze(
                jnp.take_along_axis(probs, jnp.expand_dims(sampled_ids, -1), -1), -1)
            selected_probs = jnp.where(target_mask,
                                    selected_probs, 2.)
            sorted_selected_probs = jnp.sort(selected_probs, axis=-1)
            masked_len = jnp.expand_dims(
                jnp.ceil(jnp.sum(target_mask, axis=1) * masked_ratio).astype('int32'),1)
            cut_off = jnp.take_along_axis(sorted_selected_probs, masked_len, axis=-1)
            sampled_ids = jnp.where(selected_probs < cut_off, 3, sampled_ids)

            return State(
                cur_index=state.cur_index+1,
                cur_seqs=sampled_ids,
                rng=rng,
                final_seqs=final_seqs)
        final_state = lax.while_loop(loop_cond_fn, loop_body_fn, init_state)
        return final_state.final_seqs  
        
    def tokens_to_logits(self, masked_inputs):
        logits = self.model.apply_fn({'params': self.model.params}, input_ids=masked_inputs, labels=None)
        return logits  
    
    def render(self,
               layout,
               offset,
               canvas_w=255, 
               canvas_h=300):
        
        no_elements = int(np.count_nonzero(layout) / 5)
        print("This layout has {} elements.".format(no_elements))
        layout = np.where(layout==0, layout, layout-offset)
        layout = np.reshape(layout, (-1, 5))
        
        canvas = Image.new("RGB", (canvas_w, canvas_h), self.color_map["background"])
        draw = ImageDraw.Draw(canvas)
        for element in layout[:no_elements]:
            color = self.color_map[self.id_to_label[element[0]]]
            center_x, center_y, width, height = element[3], element[4], element[1], element[2]
            min_x = np.round(center_x - width/2. + 1e-7)
            max_x = np.round(center_x + width/2. + 1e-7)
            min_y = np.round(center_y - height/2. + 1e-7)
            max_y = np.round(center_y + height/2. + 1e-7)
            
            min_x = round(np.clip(min_x/(self.resolution_w-1), 0., 1.) * canvas_w)
            min_y = round(np.clip(min_y/(self.resolution_h-1), 0., 1.) * canvas_h)
            max_x = round(np.clip(max_x/(self.resolution_w-1), 0., 1.) * canvas_w)
            max_y = round(np.clip(max_y/(self.resolution_h-1), 0., 1.) * canvas_h)

            draw.rectangle([min_x, min_y, max_x, max_y],
                           outline=color,
                           fill=None, 
                           width=1)
        
        legend_width = 200
        legend_height = len(self.id_to_label) * 30  # Adjust as needed
        legend = Image.new("RGB", (legend_width, legend_height), (100,100,100))
        legend_draw = ImageDraw.Draw(legend)
        for i, mapping in enumerate(self.id_to_label):
            class_name = self.id_to_label[mapping]
            color = self.color_map[class_name]
            # Draw legend entry rectangle with the class color
            legend_entry_y = i * 30
            legend_draw.rectangle([10, legend_entry_y, 30, legend_entry_y + 20], outline=color, fill=color, width=1)

            # Draw legend entry text with the class name
            legend_font = ImageFont.load_default()
            legend_text_x = 40
            legend_text_y = legend_entry_y + 5
            legend_draw.text((legend_text_x, legend_text_y), class_name, fill="black", font=legend_font)

        # Combine the main canvas and legend side by side
        combined_canvas = Image.new("RGB", (canvas_w + legend_width, canvas_h), (100,100,100))
        combined_canvas.paste(canvas, (0, 0))
        combined_canvas.paste(legend, (canvas_w, canvas_h - legend_height))
        display(combined_canvas)

    def render_two_layouts(self,
                           ground_truth,
                           generated,
                           mask,
                           offset,
                           canvas_w=255, 
                           canvas_h=300):
        no_elements = int(np.count_nonzero(generated) / 5)
        print("This layout has {} elements.".format(no_elements))
        position_ids = jnp.arange(ground_truth.shape[-1])[None, :]
        is_asset = position_ids % 5 == 0
        is_size = functools.reduce(
            lambda x,y: x|y,
            [position_ids % 5 == i for i in range(1, 3)])
        is_position = functools.reduce(
            lambda x,y: x|y,
            [position_ids % 5 == i for i in range(3, 5)])
        is_mask_asset = jnp.sum(mask * is_asset)
        if is_mask_asset:
            print("{} masked token are asset".format(is_mask_asset))
        is_mask_position = jnp.sum(mask * is_position) 
        if is_mask_position:
            print("{} masked tokens are center positions".format(is_mask_position))
        is_mask_size = jnp.sum(mask * is_size)
        if is_mask_size:
            print("{} masked tokens are sizes".format(is_mask_size))

        generated = np.where(generated==0, generated, generated-offset)
        generated = np.reshape(generated, (-1, 5))
        ground_truth = np.where(ground_truth==0, ground_truth, ground_truth-offset)
        ground_truth = np.reshape(ground_truth, (-1, 5))

        canvas1 = Image.new("RGB", (canvas_w, canvas_h), self.color_map["background"])
        canvas1_draw = ImageDraw.Draw(canvas1)
        for element in ground_truth[:no_elements]:
            color = self.color_map[self.id_to_label[element[0]]]
            center_x, center_y, width, height = element[3], element[4], element[1], element[2]
            min_x = np.round(center_x - width/2. + 1e-4)
            max_x = np.round(center_x + width/2. + 1e-4)
            min_y = np.round(center_y - height/2. + 1e-4)
            max_y = np.round(center_y + height/2. + 1e-4)
            
            min_x = round(np.clip(min_x/(self.resolution_w-1), 0., 1.) * canvas_w)
            min_y = round(np.clip(min_y/(self.resolution_h-1), 0., 1.) * canvas_h)
            max_x = round(np.clip(max_x/(self.resolution_w-1), 0., 1.) * canvas_w)
            max_y = round(np.clip(max_y/(self.resolution_h-1), 0., 1.) * canvas_h)

            canvas1_draw.rectangle([min_x, min_y, max_x, max_y],
                           outline=color,
                           fill=None, 
                           width=1)
        
        canvas2 = Image.new("RGB", (canvas_w, canvas_h), self.color_map["background"])
        canvas2_draw = ImageDraw.Draw(canvas2)
        for element in generated[:no_elements]:
            color = self.color_map[self.id_to_label[element[0]]]
            center_x, center_y, width, height = element[3], element[4], element[1], element[2]
            min_x = np.round(center_x - width/2. + 1e-4)
            max_x = np.round(center_x + width/2. + 1e-4)
            min_y = np.round(center_y - height/2. + 1e-4)
            max_y = np.round(center_y + height/2. + 1e-4)
            
            min_x = round(np.clip(min_x/(self.resolution_w-1), 0., 1.) * canvas_w)
            min_y = round(np.clip(min_y/(self.resolution_h-1), 0., 1.) * canvas_h)
            max_x = round(np.clip(max_x/(self.resolution_w-1), 0., 1.) * canvas_w)
            max_y = round(np.clip(max_y/(self.resolution_h-1), 0., 1.) * canvas_h)

            canvas2_draw.rectangle([min_x, min_y, max_x, max_y],
                           outline=color,
                           fill=None, 
                           width=1)

        legend_width = 200
        legend_height = len(self.id_to_label) * 30  # Adjust as needed
        legend = Image.new("RGB", (legend_width, legend_height), (100,100,100))
        legend_draw = ImageDraw.Draw(legend)
        for i, mapping in enumerate(self.id_to_label):
            class_name = self.id_to_label[mapping]
            color = self.color_map[class_name]
            # Draw legend entry rectangle with the class color
            legend_entry_y = i * 30
            legend_draw.rectangle([10, legend_entry_y, 30, legend_entry_y + 20], outline=color, fill=color, width=1)

            # Draw legend entry text with the class name
            legend_font = ImageFont.load_default()
            legend_text_x = 40
            legend_text_y = legend_entry_y + 5
            legend_draw.text((legend_text_x, legend_text_y), class_name, fill="black", font=legend_font)

        # Combine the main canvas and legend side by side
        combined_canvas = Image.new("RGB", (2*canvas_w + legend_width, canvas_h), (100,100,100))
        combined_canvas.paste(canvas1, (0, 0))
        combined_canvas.paste(canvas2, (canvas_w, 0))
        combined_canvas.paste(legend, (2*canvas_w, canvas_h - legend_height))
        display(combined_canvas)

    def generate_from_layout(self, input, offset, possible_logit,
                             input_mask_token=-1, output_mask_token=3, 
                             pad_token=0, canvas_w=255, canvas_h=300):
        seq_len = offset.shape[-1]
        input = np.array(input)
        offset_subset = np.array(offset[0][:len(input)])
        input = np.where(input!=input_mask_token,
                         input+offset_subset,
                         output_mask_token)
        input = np.pad(input, (0, seq_len-len(input)), constant_values=pad_token)
        input = np.expand_dims(input, axis=0)
        generated = self.decode(model = None, inputs = input, logit_masks = possible_logit)[0][-1]
        self.render(generated, offset, canvas_w=canvas_w, canvas_h=canvas_h)
        return generated
    
    def generate_from_layout_to_json(self, input, offset, possible_logit,
                                     input_mask_token=-1, output_mask_token=3,
                                     pad_token=0, canvas_w=255, canvas_h=300):

        total_dim = self.layout_dim*2 + 1
        no_element = len(input) // total_dim

        seq_len = offset.shape[-1]
        input = np.array(input)
        offset_subset = np.array(offset[0][:len(input)])
        input = np.where(input!=input_mask_token,
                         input+offset_subset,
                         output_mask_token)
        input = np.pad(input, (0, seq_len-len(input)), constant_values=pad_token)
        input = np.expand_dims(input, axis=0)
        
        generated = self.decode(model = None, inputs = input, logit_masks = possible_logit)[0][-1]
        generated = np.where(generated!=pad_token,
                            generated - offset[0],
                            input_mask_token)
        generated = np.reshape(generated, (-1, 5))     
        ele_list = []
        for e in generated[:no_element]:
            ele_dict = {}
            ele_dict["class"] = self.id_to_label[e[0]]
            
            center_x, center_y, width, height = e[3], e[4], e[1], e[2]
            min_x = np.round(center_x - width/2. + 1e-4)
            min_y = np.round(center_y - height/2. + 1e-4)
            min_x = round(np.clip(min_x/(self.resolution_w-1), 0., 1.) * canvas_w)
            min_y = round(np.clip(min_y/(self.resolution_h-1), 0., 1.) * canvas_h)
            width = round(np.clip(width/(self.resolution_w-1), 0., 1.) * canvas_w)
            height = round(np.clip(height/(self.resolution_h-1), 0., 1.) * canvas_h)
            
            ele_dict["x"] = min_x
            ele_dict["y"] = min_y
            ele_dict["width"] = width
            ele_dict["height"] = height

            ele_list.append(ele_dict)

        return ele_list