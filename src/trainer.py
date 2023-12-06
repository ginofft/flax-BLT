import jax, flax, wandb, time, random, functools
from jax import jit
import jax.numpy as jnp
from flax.training import train_state, orbax_utils, common_utils
import optax, orbax
from clu import metrics
from torch.utils.data import DataLoader
from livelossplot import PlotLosses
import numpy as np

from functools import partial
from typing import Optional
from pathlib import Path

from .dataset import LayoutDataset, PaddingCollator
from .decoder import LayoutDecoder
from .models.biodirectional_layout import BLT
from .utils import attribute_random_masking, is_notebook, attribute_size_position_masking

@flax.struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")

@flax.struct.dataclass
class TrainState(train_state.TrainState):
    metrics: Metrics

class BERTLayoutTrainer:
    """BERT-style Layout Trainer - one stop class for training, inferencing

    Attributes
    ----------
    config : ml_collections.ConfigDict
        Config dictionary for trainer, storing all necessary parameters and path.
    workdir : str
        Path to folder,  in which checkpoint would be saved in.
    layout_dim : int
        The dimension of the layout. Usually 2 - meaning 2D.
    rng : Union[Array, PNGKeyArray]
        JAX's pseduorandom random number generator.
    layout_dim : int
        The dimensionality of the Layout - usually 2 (2d) but can be 3(3d)
    total_dim : int
        The no. dimension of a layout's element.
    
    Methods
    -------
    create_train_state() -> TrainState: 
        return a FLAX's TrainState, composing of: parameters, optimizer and metrics
    train() -> None:
        train a model and save into self.workdir. DOES NOT return anything
    test() -> None: 
        NotDocumented.
    preprocess_batch() -> tuple:
        process batch to model's input format. Mainly used to create [MASK] tokens.
    """

    def __init__(self, config, workdir: Optional[str] = None, wandb=True):
        self.config = config
        if workdir is not None:
            self.workdir = Path(workdir)
            if self.workdir.exists() == False:
                raise Exception(f"{workdir} does not exist!")
        else:
            self.workdir = Path('')
        self.rng = jax.random.PRNGKey(config.seed)
        self.dtype, self.data_dtype = self._get_dtype()
        self.layout_dim = self.config.layout_dim
        self.total_dim = self.layout_dim*2 + 1
        if wandb:
            wandb.init(project='BLT', config=config,
                    name=str(int(time.time()))+'_'+config.name)

    def preprocess_batch(self, batch, batch_size):
        label = None
        if batch.shape[0] != batch_size:
            return None, None
        batch = attribute_size_position_masking(
            inputs=batch, mask_token=3,
            pad_token=0, layout_dim=self.layout_dim)
        return batch, label
    
    def create_train_state(
            self,
            rng,
            inputs,
            vocab_size):
        model = BLT(use_vertical=self.config.use_vertical_info,
                    vocab_size=vocab_size,
                    hidden_size=self.config.qkv_dim,
                    num_hidden_layers=self.config.num_layers,
                    num_attention_heads=self.config.num_heads,
                    intermediate_size=self.config.mlp_dim,
                    hidden_dropout_prob=self.config.dropout_rate,
                    attention_dropout_prob=self.config.attention_dropout_rate,
                    asset_position_embedding=self.config.sequential_embedding,
                    pad_token_id=0,
                    layout_dim=self.config.layout_dim)
        param_rng, dropout_rng, rng = jax.random.split(rng, 3)
        params = model.init({
                "params": param_rng,
                "dropout": dropout_rng
            },
            input_ids=inputs["inputs"],
            labels=inputs["labels"],
            deterministic=False
        )["params"]

        lr_fn = self._create_lr_scheduler(lr = self.config.optimizer.lr, 
                                          warmup_step=self.config.optimizer.warmup_steps)
        tx = self._create_optimizer(lr_fn)
        
        return TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
            metrics=Metrics.empty()
        )
    
    def train(self):
        if is_notebook():
            liveplot = PlotLosses()
        else:
            liveplot = None

        #Setup Dataset and DataLoader
        train_dataset = LayoutDataset('train',
                                      self.config.dataset_path+'/train.json',
                                      config=self.config.dataset)
        val_dataset = LayoutDataset('validation',
                                    self.config.dataset_path+'/val.json',
                                    config=self.config.dataset)
        collator = PaddingCollator(pad_token_id=train_dataset.pad_idx, seq_len=train_dataset.seq_len)
        
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size,
                                      collate_fn=collator.collate_padding,
                                      shuffle=self.config.train_shuffle,
                                      drop_last=False)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config.eval_batch_size,
                                    collate_fn=collator.collate_padding,
                                    shuffle=self.config.train_shuffle,
                                    drop_last=False)
        # Setup model's state
        init_batch = jnp.ones((self.config.batch_size, train_dataset.seq_len))
        init_label = jnp.ones((self.config.batch_size, 1))
        init_batch = dict(inputs=init_batch, labels=init_label)
        vocab_size = train_dataset.get_vocab_size()
        state = self.create_train_state(rng = self.rng, inputs = init_batch, vocab_size=vocab_size)

        if self.config.checkpoint_path is not None:
            target = {'model':state, 'metric_history':dict, 'min_loss': float, 'epoch':int}
            ckpt = self._load_checkpoint(target)
            state = ckpt['model']

            metric_history = ckpt['metric_history']
            min_validation_loss = ckpt['min_loss']
            start_epoch = ckpt['epoch']

            print("Loaded epoch {} - Loss: {:.6f}".format(start_epoch, min_validation_loss))
        else:
            metric_history = {'loss': [],
                            'val_loss': []}    
            min_validation_loss = float('inf')
            start_epoch = 0
            print("Start training from scratch")

        # Get possible logit for each position
        pos_info = [[train_dataset.offset_class, train_dataset.number_classes], 
                    [train_dataset.offset_width, train_dataset.resolution_w],
                    [train_dataset.offset_height, train_dataset.resolution_h],
                    [train_dataset.offset_x, train_dataset.resolution_w],
                    [train_dataset.offset_y, train_dataset.resolution_h]]
        seq_len = train_dataset.seq_len
        possible_logit, _ = self._make_possible_mask(vocab_size=vocab_size, pos_info=pos_info,seq_len=seq_len)
        self.rng, train_rng = jax.random.split(self.rng, 2)
        # Training / Validation Loop
        for epoch in range(start_epoch+1, self.config.epoch+1):
            # Train
            for batch in train_dataloader:
                step_rng = jax.random.fold_in(train_rng, state.step)
                batch = attribute_size_position_masking(batch, mask_token=train_dataset.mask_idx,
                                                pad_token=train_dataset.pad_idx, layout_dim=self.config.layout_dim)
                state = self._train_step(state = state, 
                                    batch = batch,
                                    rng = step_rng,
                                    possible_mask = possible_logit)
                state = self._compute_metrics(state = state, 
                                              batch = batch,
                                              possible_mask = possible_logit)
            for metric, value in state.metrics.compute().items():
                metric_history['loss'].append(value)
            state = state.replace(metrics=state.metrics.empty())
            
            #Validate
            for batch in val_dataloader:
                batch = attribute_size_position_masking(batch, mask_token=train_dataset.mask_idx,
                                                 pad_token=train_dataset.pad_idx, layout_dim=self.config.layout_dim)
                validation_state = self._compute_metrics(state = state, 
                                                         batch=batch,
                                                         possible_mask = possible_logit)      
            for metric, value in validation_state.metrics.compute().items():
                metric_history['val_loss'].append(value)
            state = state.replace(metrics=state.metrics.empty())
            
            # Log messages
            logs = {}
            logs["loss"] = metric_history["loss"][-1]
            logs["val_loss"] = metric_history["val_loss"][-1]
            if liveplot:
                liveplot.update(logs)
                liveplot.send()
            else:
                print('Epoch {} train/val loss: {:.6f} / {:.6f}'.format(epoch,
                                                                        logs['loss'],
                                                                        logs['val_loss']))
            wandb.log({
                'train_loss': metric_history["loss"][-1],
                'val_loss': metric_history["val_loss"][-1]
            }, step=epoch)

            validation_loss = metric_history['val_loss'][-1]
            if validation_loss < min_validation_loss:
                min_validation_loss = validation_loss
                ckpt = {'model': state, 'epoch':epoch, 
                        'metric_history': metric_history, 'min_loss': min_validation_loss}
                self._save_checkpoint(ckpt, 'best')
            if epoch % self.config.save_every_epoch == 0:
                ckpt = {'model': state, 'epoch':epoch, 
                        'metric_history': metric_history, 'min_loss': min_validation_loss}
                self._save_checkpoint(ckpt, f'checkpoint_epoch{epoch}')
                self._sample_layout(epoch, state, val_dataloader)
        return min_validation_loss
    
    def test(self):
        raise NotImplementedError
    
    @partial(jit, static_argnums=(0,))
    def _train_step(self, state, batch, rng, possible_mask=None):
        def loss_fn(params):
            logits = state.apply_fn({'params':params}, 
                                    input_ids=batch["masked_inputs"], 
                                    labels=None, 
                                    rngs={"dropout":rng})
            #loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            loss = self._compute_weighted_cross_entropy(logits, batch["targets"], batch["weights"], possible_mask)
            return loss
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state
    
    @partial(jit, static_argnums=(0,))
    def _compute_metrics(self,
                         state, 
                         batch, 
                         possible_mask = None):
        logits = state.apply_fn({'params': state.params}, input_ids=batch['masked_inputs'], labels=None, deterministic=True)
        loss = self._compute_weighted_cross_entropy(logits, batch["targets"], batch["weights"], possible_mask)
        metric_updates = state.metrics.single_from_model_output(loss = loss)
        metrics = state.metrics.merge(metric_updates)
        state = state.replace(metrics=metrics)
        return state
    
    @partial(jit, static_argnums=(0,))
    def _compute_weighted_cross_entropy(self,
                                        logits,
                                        targets,
                                        mask=None,
                                        logit_mask=None,
                                        label_smoothing=0.0):
        """Negative Log-likelihood for BLT modelling

        Basically log_cross_entropy(), but with special logit_mask for possible tokens

        Arguments
        ---------
        logits : float jnp.array 
            [batch_size, seq_len, vocab_size] model's output logits
        targets : int jnp.array
            [batch_size, seq_len] integer label (NOT onehot)
        mask : int jnp.array
            [batch_size, seq_len] Whether or not an element is replace by [MASK] token
        logit_mask : int jnp.array
            [batch_size, seq_len, vocab_size] mask of possible logit
        label_smoothing : float
            label smoothing constant. DO NOT TOUCH unless you know what you are doing.
        """
        logits = jnp.where(logit_mask>0, -1e7, logits)

        vocab_size = logits.shape[-1]
        confidence = 1.0 - label_smoothing
        low_confidence = label_smoothing / (vocab_size-1)
        soft_targets = common_utils.onehot(targets, vocab_size,
                                            on_value=confidence, off_value=low_confidence)
        loss = -jnp.sum(soft_targets * flax.linen.log_softmax(logits), axis=-1)
        normalizing_constant = -(
            confidence * jnp.log(confidence) +
            (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))
        loss = loss - normalizing_constant
        normalizing_factor = np.prod(targets.shape)
        if mask is not None:
            loss = loss * mask
            normalizing_factor = mask.sum()
        return loss.sum() / normalizing_factor

    def _get_dtype(self):
        if self.config.dtype == "bfloat16":
            return jnp.bfloat16, jnp.bfloat16
        else:
            return jnp.float32, jnp.float32
    
    def _make_possible_mask(self, vocab_size, pos_info, seq_len):
        """Create masking for possible logits at each position.
        
        For example, our sequence might have the form of [c1,w1,h1,x1,y2,c2,w2, ...]. Which meant
        that as the [c1,c2,...,c_n] position, only token corresonpind to element class are available.
        As such, it is benefitial to mask tokens at each position to a subset of the vocabulary for
        faster training.

        Arguments
        ---------
        vocab_size : int
            the number of possible vocabulary
        pos_info : list[[int, int]]
            start token index and the number of possible index for a type of token. For example,
            Element's class token begin at 4 - with 4 possible tokens, and Element's width begin at 8,
            with 32 possible tokens.The pos_info for them would be [[4,4],[8,32]] 
        seq_len : int
            the length of the sequence
            
        Returns
        -------
        asset_logit_masks: jnp.array
            [1, seq_len, vocab_size] logits mask for each position
        asset_offset: jnp.array
            [1, seq_len] offset for each position.
        """
        total_dim = self.total_dim
        logit_masks = []

        offset = jnp.array([pi[0] for pi in pos_info])
        offset = jnp.expand_dims(offset, 0)
        # tile offset to seq_len (which should be divisible by total_dim)
        asset_offset = jnp.tile(offset, (1, seq_len//total_dim))
        # Create possible mask for each type of token 
        for idx, pi in enumerate(pos_info):
            # Create initial impossible token mask ( 0 = possible, 1 = impossible)
            logit_mask = jnp.ones((1,1,vocab_size))
            # Create possible tokens
            pos_mask = jnp.zeros((1,1,pi[1]))
            # Add posible and impossible mask together
            logit_mask = jax.lax.dynamic_update_slice(logit_mask, 
                                                      pos_mask,
                                                      (0, 0, pi[0]))
            # --> For example, at position [c1], the possible token should be [1, 1, 1, 1, 0, 0, 0, 0, 1, ...1]
            # denoting that only token representing element class is usuable
            # if idx == 0:
            #     logit_mask = logit_mask.at[:,:,2].set(0)
            logit_masks.append(logit_mask)
        
        logit_masks = jnp.concatenate(logit_masks, axis=1)
        asset_logit_masks = jnp.tile(logit_masks, (1, seq_len//total_dim, 1))
        if seq_len % total_dim > 0:
            asset_logit_masks = jnp.concatenate(
                (asset_logit_masks, logit_masks[:, :(seq_len % total_dim), :]),
                axis=1)
            asset_offset = jnp.concatenate(
                (asset_offset, offset[:, :(seq_len % total_dim)]), axis=1)
        return asset_logit_masks, asset_offset
    
    def _create_optimizer(self, lr_fn):
        opt_def = optax.adamw(
            learning_rate=lr_fn,
            b1=self.config.optimizer.beta1,
            b2=self.config.optimizer.beta2,
            weight_decay=self.config.optimizer.weight_decay
        )
        return opt_def
    
    def _create_lr_scheduler(self, lr=1, warmup_step=4000):
        def step_fn(step):
            cur_lr = lr * jnp.minimum(1.0, step/warmup_step) / jnp.sqrt(
                jnp.maximum(step, warmup_step))
            return jnp.asarray(cur_lr, dtype=jnp.float32)
        return step_fn
    
    def _load_checkpoint(self, target):
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint_path = Path(self.config.checkpoint_path)
        if checkpoint_path.exists() == False:
            raise Exception(f"{checkpoint_path} does not exits!!")
        ckpt = orbax_checkpointer.restore(checkpoint_path, item=target)
        return ckpt

    def _save_checkpoint(self, ckpt, name):
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(self.workdir/name, ckpt, save_args=save_args, force=True)

    def _sample_layout(self, epoch, model, dataloader):
        # Get based layout 
        dataset = dataloader.dataset   
        for sample_batch in dataloader:
            layout = sample_batch[:4] # get 4 random design
            break

        # True/False Maskings
        position_ids = jnp.arange(layout.shape[-1])[None, :]
        is_pad = layout == dataset.pad_idx
        is_asset = position_ids % 5 == 0 # [1, seq_len]
        is_size = functools.reduce(
            lambda x,y: x|y,
            [position_ids % 5 == i for i in range(1, 3)]) # [1, seq_len] 
        is_position = functools.reduce(
            lambda x,y: x | y,
            [position_ids % 5 == i for i in range(3, 5)]) # [1, seq_len] 

        # Get Decoder 
        decoder = LayoutDecoder(dataset.get_vocab_size(), dataset.seq_len, 
                                self.config.layout_dim, dataset.ID_TO_LABEL,
                                dataset.COLORS, iterative_nums=np.array([1,7,7]), 
                                temperature=1.0)

        # C -> S + P
        mask = (is_asset) | (is_pad)
        asset_layout = jnp.where(mask, layout, dataset.mask_idx)

        # C + S -> P
        mask = (is_position) & (~is_pad) 
        asset_size_layout = jnp.where(mask, dataset.mask_idx, layout)

        # C + S(first element) -> S(other elements) + P
        mask = ((is_asset) | (is_pad)) | ((is_size) & (position_ids<5))
        asset_firstSize_layout = jnp.where(mask, layout, dataset.mask_idx)

        pos_info = [[dataset.offset_class, dataset.number_classes], 
                    [dataset.offset_width, dataset.resolution_w],
                    [dataset.offset_height, dataset.resolution_h],
                    [dataset.offset_x, dataset.resolution_w],
                    [dataset.offset_y, dataset.resolution_h]]
        possible_logit, _ = self._make_possible_mask(vocab_size=dataset.get_vocab_size(),
                                                     pos_info=pos_info, 
                                                     seq_len=dataset.seq_len)
        
        seq_asset_layout = decoder.decode(model, asset_layout, possible_logit)
        seq_size_layout = decoder.decode(model, asset_size_layout, possible_logit)
        seq_asset_firstSize_layout= decoder.decode(model, asset_firstSize_layout, possible_logit)
        
        render_layout = [dataset.render(l) for l in layout]
        render_asset_layout = [dataset.render(np.array(l[-1], copy=False)) for l in seq_asset_layout]
        render_size_layout = [dataset.render(np.array(l[-1], copy=False)) for l in seq_size_layout]
        render_asset_firstPos_layout = [dataset.render(np.array(l[-1], copy=False)) for l in seq_asset_firstSize_layout]

        wandb.log({
            "base_layouts": [wandb.Image(pil, caption='input_{:02d}_{:02d}.png'.format(epoch,i))
                                for i, pil in enumerate(render_layout)],
            "C->S+P layouts": [wandb.Image(pil, caption='recon_{:02d}_{:02d}.png'.format(epoch,i))
                                for i, pil in enumerate(render_asset_layout)],
            "C+S->P layouts": [wandb.Image(pil, caption='sample_random_{:02d}_{:02d}.png'.format(epoch,i))
                                        for i, pil in enumerate(render_size_layout)],
            "C+S(first) -> S(remain)+P layout": [wandb.Image(pil, caption='sample_det_{:02d}_{:02d}.png'.format(epoch,i))
                                    for i, pil in enumerate(render_asset_firstPos_layout)],
        }, step=epoch)
