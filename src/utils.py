import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
import frozendict

import functools

def is_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            raise ImportError("console")
            return False
    except:
        return False
    else:  # pragma: no cover
        return True
    
def convert_json_to_model_input(layout_json, label_to_id,
                                resolution_w, resolution_h, 
                                mask_token=-1):
	sorted_elements = sorted(layout_json['elements'], key=lambda x: label_to_id.get(x['class_name'], len(label_to_id)))
	layout_json['elements'] = sorted_elements
	canvas_w = layout_json["width"]
	canvas_h = layout_json["height"]
	input = []
	for e in layout_json["elements"]:
		element_input = [-1,-1,-1,-1,-1]
		if e["class_name"]:
			class_idx = label_to_id[e["class_name"]]
			element_input[0] = class_idx
		center_x, center_y = e["center_x"], e["center_y"]
		width, height = e["width"], e["height"]
		if center_x:
			center_x = center_x / canvas_w
			discrete_x = round(center_x * (resolution_w - 1))
			element_input[3] = discrete_x
		if center_y:
			center_y = center_y / canvas_h
			discrete_y = round(center_y * (resolution_h - 1))
			element_input[4] = discrete_y
		if width:
			width = width / canvas_w
			discrete_width = round(
				np.clip(width * (resolution_w - 1), 1., resolution_w-1))
			element_input[1] = discrete_width
		if height:
			height = height / canvas_h
			discrete_height = round(
				np.clip(height * (resolution_h - 1), 1., resolution_h-1))
			element_input[2] = discrete_height
		input.extend(element_input)
	return input
    
def attribute_random_masking(inputs, mask_token, pad_token, layout_dim):
	"""Repace some token with [mask] token. 
	However, for a specific layout, only token with the same semantic meaning are masked.
	"""
	targets = inputs
	total_dim = layout_dim*2 + 1
	rng = jax.random.PRNGKey(jnp.sum(inputs, dtype='int32'))
		
	# Get positions (in the tokenized sequence)
	# - [PAD] token position
	# - element's class position
	# - element's width height position
	# - element's center position
	is_pad = inputs == pad_token # [batch size, seq_len]
	position_ids = jnp.arange(inputs.shape[-1])[None, :]
	is_asset = position_ids % total_dim == 0 # [1, seq_len]
	is_size = functools.reduce(
		lambda x,y: x|y,
		[position_ids % total_dim == i for i in range(1, layout_dim+1)]) # [1, seq_len] 
	is_position = functools.reduce(
		lambda x,y: x | y,
		[position_ids % total_dim == i for i in range(layout_dim+1, total_dim)]) # [1, seq_len] 
	rand = jax.random.uniform(rng, (inputs.shape[0],1)) # [batch size, 1]
	# There dimension are as they are to support forecasting 
	pre_masked_inputs = inputs.copy()
	
	# all token are [MASKED]
	fullmask = jnp.full_like(inputs, mask_token)
	fullmask = jnp.where(is_pad, pad_token, fullmask) # [PAD] token 
	
	# Whether to mask asset, size or position
	weights = is_asset & (~is_pad) # Mask every asset
	weights = jnp.where(jnp.logical_and(rand >= 0.33, rand < 0.67), is_size & (~is_pad), weights) # if [condition], mask size
	weights = jnp.where(rand >= 0.67, is_position & (~is_pad), weights) # if [condition], mask position
	
	lens = jnp.sum(weights, axis=-1) # [batch size] , no. POSSIBLE mask token per layout
	rng, subrng = jax.random.split(rng)
	mask_rate = 1 - jax.random.uniform(subrng, lens.shape) # [batch size], mask rate of each layout

	mask_lens = jax.lax.ceil(lens*mask_rate).astype('int32') # [batch size], no. ACTUAL mask token per layout
	rng, subrng = jax.random.split(rng) 
	
	should_mask = jax.random.uniform(subrng, shape=inputs.shape) # [batch size, seq_len], probability of being masked
	# If weights is true (can be mask for this layout), keep the probability, else equal 2 -> prevent being masked
	should_mask = jnp.where(weights, should_mask, 2.) # [batch size, seq_len]
	sorted_should_mask = jnp.sort(should_mask, axis=-1) # [batch size, seq_len] sorted version
	cut_off = jnp.take_along_axis(
		sorted_should_mask, jnp.expand_dims(mask_lens-1, 1), axis=-1) # [batch size, 1] the cut off of each layout
	should_mask = jnp.where(should_mask <= cut_off, 1, 0) # [batch size, seq_len] where there should be masked

	masked_inputs = jnp.where(should_mask, fullmask, pre_masked_inputs) # [batch size, seq_len] masked input
	weights = jnp.where(is_pad, 0, should_mask) 
	return dict(masked_inputs=masked_inputs, targets=targets, weights=weights)

def attribute_size_position_masking(inputs, mask_token, pad_token, layout_dim):
	"""Repace some token with [mask] token. 
	However, for a specific layout, only token with the same semantic meaning are masked.
	We only mask position and size here.
	"""
	targets = inputs
	total_dim = layout_dim*2 + 1
	rng = jax.random.PRNGKey(jnp.sum(inputs, dtype='int32'))
		
	# Get positions (in the tokenized sequence)
	# - [PAD] token position
	# - element's class position
	# - element's width height position
	# - element's center position
	is_pad = inputs == pad_token # [batch size, seq_len]
	position_ids = jnp.arange(inputs.shape[-1])[None, :]
	is_asset = position_ids % total_dim == 0 # [1, seq_len]
	is_size = functools.reduce(
		lambda x,y: x|y,
		[position_ids % total_dim == i for i in range(1, layout_dim+1)]) # [1, seq_len] 
	is_position = functools.reduce(
		lambda x,y: x | y,
		[position_ids % total_dim == i for i in range(layout_dim+1, total_dim)]) # [1, seq_len] 
	rand = jax.random.uniform(rng, (inputs.shape[0],1)) # [batch size, 1]
	# There dimension are as they are to support forecasting 
	pre_masked_inputs = inputs.copy()
	
	# all token are [MASKED]
	fullmask = jnp.full_like(inputs, mask_token)
	fullmask = jnp.where(is_pad, pad_token, fullmask) # [PAD] token 
	
	# Whether to mask asset, size or position
	weights = (~is_pad) # Mask every asset
	weights = jnp.where(rand <= 0.5, is_size & (~is_pad), weights) # if [condition], mask size
	weights = jnp.where(rand > 0.5, is_position & (~is_pad), weights) # if [condition], mask position
	
	lens = jnp.sum(weights, axis=-1) # [batch size] , no. POSSIBLE mask token per layout
	rng, subrng = jax.random.split(rng)
	mask_rate = 1 - jax.random.uniform(subrng, lens.shape) # [batch size], mask rate of each layout

	mask_lens = jax.lax.ceil(lens*mask_rate).astype('int32') # [batch size], no. ACTUAL mask token per layout
	rng, subrng = jax.random.split(rng) 
	
	should_mask = jax.random.uniform(subrng, shape=inputs.shape) # [batch size, seq_len], probability of being masked
	# If weights is true (can be mask for this layout), keep the probability, else equal 2 -> prevent being masked
	should_mask = jnp.where(weights, should_mask, 2.) # [batch size, seq_len]
	sorted_should_mask = jnp.sort(should_mask, axis=-1) # [batch size, seq_len] sorted version
	cut_off = jnp.take_along_axis(
		sorted_should_mask, jnp.expand_dims(mask_lens-1, 1), axis=-1) # [batch size, 1] the cut off of each layout
	should_mask = jnp.where(should_mask <= cut_off, 1, 0) # [batch size, seq_len] where there should be masked

	masked_inputs = jnp.where(should_mask, fullmask, pre_masked_inputs) # [batch size, seq_len] masked input
	weights = jnp.where(is_pad, 0, should_mask) 
	return dict(masked_inputs=masked_inputs, targets=targets, weights=weights)
	
def get_magazine_config():

	"""Gets the default hyperparameter configuration."""
	config = ml_collections.ConfigDict()
  	# Exp info
	config.checkpoint_path = None
	config.dataset_path = "data/layoutdata/json"
	config.experiment = "bert_layout"
	config.model_class = "bert_layout"
	config.image_size = 256

	# Training info
	config.epoch = 500
	config.layout_dim = 2
	config.seed = 56175
	config.batch_size = 64
	config.train_shuffle = True
	config.eval_pad_last_batch = False
	config.eval_batch_size = 64
	config.save_every_epoch = 50

	# Model info
	config.dtype = "float32"
	config.autoregressive = False
	config.sequential_embedding = True
	config.shuffle_buffer_size = 10
	config.use_vae = True
	config.share_embeddings = True
	config.num_layers = 4
	config.qkv_dim = 512
	config.mlp_dim = 2048
	config.num_heads = 8
	config.dropout_rate = 0.1
	config.attention_dropout_rate = 0.3
	config.label_smoothing = 0.
	config.sampling_method = "top-p"
	config.use_vertical_info = False

	# Optimizer info
	config.optimizer = ml_collections.ConfigDict()
	config.optimizer.warmup_steps = 4000
	config.optimizer.lr = 5e-3
	config.optimizer.beta1 = 0.9
	config.optimizer.beta2 = 0.98
	config.optimizer.weight_decay = 0.01
	config.beta_rate = 1 / 20_000

	# Dataset info
	config.dataset = ml_collections.ConfigDict()
	config.dataset.LABEL_NAMES = ("text", "image", "text-over-image", "headline", "headline-over-image")
	config.dataset.COLORS = {
			"text": (254, 231, 44),
			"image": (27, 187, 146),
			"headline": (255, 0, 0),
			"text-over-image": (0, 102, 255),
			"headline-over-image": (204, 0, 255),
			"background": (200, 200, 200)}
	config.dataset.FRAME_WIDTH = 225
	config.dataset.FRAME_HEIGHT = 300

	config.dataset.ID_TO_LABEL = frozendict.frozendict(
			{i: v for (i,v) in enumerate(config.dataset.LABEL_NAMES)})
	config.dataset.NUMBER_LABELS = len(config.dataset.ID_TO_LABEL)
	config.dataset.LABEL_TO_ID_ = frozendict.frozendict(
			{l: i for i,l in config.dataset.ID_TO_LABEL.items()})
		
	return config

def get_publaynet_config():
	"""Gets the default hyperparameter configuration."""

	config = ml_collections.ConfigDict()
	# Exp info
	config.name = 'publaynet'
	config.checkpoint_path = None
	config.dataset_path = "data/publaynet"
	config.experiment = "bert_layout"
	config.model_class = "bert_layout"
	config.image_size = 256

	# Training info
	config.epoch = 500
	config.seed = 0
	config.max_length = 130
	config.batch_size = 64
	config.train_shuffle = True
	config.eval_pad_last_batch = False
	config.eval_batch_size = 64
	config.save_every_epoch = 20

	# Model info
	config.layout_dim = 2
	config.dtype = "float32"
	config.autoregressive = False
	config.sequential_embedding = True
	config.shuffle_buffer_size = 10
	config.use_vae = True
	config.share_embeddings = True
	config.num_layers = 4
	config.qkv_dim = 512
	config.mlp_dim = 2048
	config.num_heads = 8
	config.dropout_rate = 0.1
	config.attention_dropout_rate = 0.3
	config.label_smoothing = 0.
	config.sampling_method = "top-p"
	config.use_vertical_info = False

	# Optimizer info
	config.optimizer = ml_collections.ConfigDict()
	config.optimizer.type = "adam"
	config.optimizer.warmup_steps = 4000
	config.optimizer.lr = 5e-3
	config.optimizer.beta1 = 0.9
	config.optimizer.beta2 = 0.98
	config.optimizer.weight_decay = 0.01
	config.beta_rate = 1 / 20_000

	# Dataset info
	config.dataset = ml_collections.ConfigDict()
	config.dataset.LABEL_NAMES = ("text", "title", "list", "table", "figure")
	config.dataset.COLORS = {
			"title": (193, 0, 0),
			"list": (64, 44, 105),
			"figure": (36, 234, 5),
			"table":  (89, 130, 213),
			"text": (253, 141, 28),
			"background": (200, 200, 200)}

	config.dataset.FRAME_WIDTH = 1050
	config.dataset.FRAME_HEIGHT = 1485

	config.dataset.ID_TO_LABEL = frozendict.frozendict({
			0: "text",
			1: "title",
			2: "list",
			3: "table",
			4: "figure",
		})
	config.dataset.NUMBER_LABELS = 5
	config.dataset.LABEL_TO_ID_ = frozendict.frozendict(
			{l: i for i,l in config.dataset.ID_TO_LABEL.items()})

	return config

def get_obello_config():
	"""Gets the default hyperparameter configuration."""

	config = ml_collections.ConfigDict()
	# Exp info
	config.name = 'obello'
	config.checkpoint_path = None
	config.dataset_path = "data/obello/all"
	config.experiment = "bert_layout"
	config.model_class = "bert_layout"
	config.image_size = 256

	# Training info
	config.epoch = 3000
	config.seed = 0
	config.max_length = 130
	config.batch_size = 64
	config.train_shuffle = True
	config.eval_pad_last_batch = False
	config.eval_batch_size = 64
	config.save_every_epoch = 50

	# Model info
	config.layout_dim = 2
	config.dtype = "float32"
	config.autoregressive = False
	config.sequential_embedding = True
	config.shuffle_buffer_size = 10
	config.use_vae = True
	config.share_embeddings = True
	config.num_layers = 2
	config.qkv_dim = 256
	config.mlp_dim = 256
	config.num_heads = 4
	config.dropout_rate = 0.1
	config.attention_dropout_rate = 0.3
	config.label_smoothing = 0.
	config.sampling_method = "greedy"
	config.use_vertical_info = False

	# Optimizer info
	config.optimizer = ml_collections.ConfigDict()
	config.optimizer.type = "adam"
	config.optimizer.warmup_steps = 4000
	config.optimizer.lr = 5e-3	
	config.optimizer.beta1 = 0.9
	config.optimizer.beta2 = 0.98
	config.optimizer.weight_decay = 0.01
	config.beta_rate = 1 / 20_000

	# Dataset info
	config.dataset = ml_collections.ConfigDict()
	config.dataset.LABEL_NAMES = ('Image', 'Headline', 'Body', 
							   	'Logo', 'CTA', 'shape')
	config.dataset.COLORS = {
			'Headline' : (193, 0, 0),
			'Body' : (0, 193, 0),
			'Logo' : (0, 0, 193),
			'Image': (128, 128, 0),
			'CTA': (0, 128, 128),
			'shape' : (128, 0, 64),
			'background': (200,200,200)
			}

	config.dataset.FRAME_WIDTH = 1050
	config.dataset.FRAME_HEIGHT = 1485

	config.dataset.ID_TO_LABEL = frozendict.frozendict({
		0: 'Image',
		1: 'Headline',
		2: 'shape',
		3: 'Body',
		4: 'Logo',
		5: 'CTA',
	})

	config.dataset.NUMBER_LABELS = 6
	config.dataset.LABEL_TO_ID_ = frozendict.frozendict(
			{l: i for i,l in config.dataset.ID_TO_LABEL.items()})
	return config