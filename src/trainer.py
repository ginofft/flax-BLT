import jax
from jax import jit
import jax.numpy as jnp
import flax
from flax.training import train_state, orbax_utils, common_utils
import optax
import orbax
from clu import metrics
from torch.utils.data import DataLoader
import numpy as np

from functools import partial
from typing import Optional
from pathlib import Path

from .dataset import LayoutDataset, PaddingCollator
from .models.biodirectional_layout import BLT
from .utils import attribute_random_masking

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
    
    Methods
    -------
    create_train_state(): 
        return a FLAX's TrainState, composing of: parameters, optimizer and metrics
    train():
        train a model and save into self.workdir. DOES NOT return anything
    test(): 
        NotDocumented.
    preprocessbatch():
        process batch to model's input format. Mainly used to create [MASK] tokens.
    """

    def __init__(self, config, workdir: Optional[str] = None):
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

    def preprocess_batch(self, batch, batch_size):
        label = None
        if batch.shape[0] != batch_size:
            return None, None
        batch = attribute_random_masking(
            inputs=batch, mask_token=3,
            pad_token=0, layout_dim=self.layout_dim)
        return batch, label
    
    def create_train_state(
            self,
            rng,
            inputs):
        model = BLT(use_vertical=self.config.use_vertical_info,
                    vocab_size=self.config.vocab_size,
                    hidden_size=self.config.qkv_dim,
                    num_hidden_layers=self.config.num_layers,
                    num_attention_heads=self.config.num_heads,
                    intermediate_size=self.config.mlp_dim,
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
        tx = self._create_optimizer()
        
        return TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
            metrics=Metrics.empty()
        )
    
    def train(self):
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
                                      shuffle=self.config.train_shuffle)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config.eval_batch_size,
                                    collate_fn=collator.collate_padding,
                                    shuffle=self.config.train_shuffle)
        # Setup model's state
        init_batch = jnp.ones((self.config.batch_size, train_dataset.seq_len))
        init_label = jnp.ones((self.config.batch_size, 1))
        init_batch = dict(inputs=init_batch, labels=init_label)
        state = self.create_train_state(rng = self.rng, inputs = init_batch)

        if self.config.checkpoint_path is not None:
            target = {'model':state, 'metric_history':dict, 'min_loss': float, 'epoch':int}
            ckpt = self._load_checkpoint(target)
            state = ckpt['model']

            metric_history = ckpt['metric_history']
            min_validation_loss = ckpt['min_loss']
            start_epoch = ckpt['epoch']

            print("Loaded epoch {} - Loss: {:.6f}".format(start_epoch, min_validation_loss))
        else:
            metric_history = {'train_loss': [],
                            'validation_loss': []}    
            min_validation_loss = float('inf')
            start_epoch = 0
            print("Start training from scratch")

        # Get possible logit for each position
        vocab_size = train_dataset.get_vocab_size()
        pos_info = [[train_dataset.offset_class, train_dataset.number_classes], 
                    [train_dataset.offset_width, train_dataset.resolution_w],
                    [train_dataset.offset_height, train_dataset.resolution_h],
                    [train_dataset.offset_center_x, train_dataset.resolution_w],
                    [train_dataset.offset_center_y, train_dataset.resolution_h]]
        seq_len = train_dataset.seq_len
        possible_logit, _ = self._make_possible_mask(vocab_size=vocab_size, pos_info=pos_info,seq_len=seq_len)
        # Training / Validation Loop
        for epoch in range(start_epoch+1, self.config.epoch+1):
            # Train
            for batch in train_dataloader:
                batch = attribute_random_masking(batch, mask_token=train_dataset.mask_idx,
                                                pad_token=train_dataset.pad_idx, layout_dim=self.config.layout_dim)
                state = self.train_step(state = state, 
                                        batch = batch,
                                        weight_mask = batch["weights"], 
                                        possible_mask = possible_logit)
                state = self.compute_metrics(state = state, 
                                             batch = batch,
                                             weight_mask = batch["weights"],
                                             possible_mask = possible_logit)
            for metric, value in state.metrics.compute().items():
                metric_history[f'train_{metric}'].append(value)
            
            #Validate
            validation_state = state
            for batch in val_dataloader:
                batch = attribute_random_masking(batch, mask_token=train_dataset.mask_idx,
                                                 pad_token=train_dataset.pad_idx, layout_dim=self.config.layout_dim)
                validation_state = self.compute_metrics(state = state, 
                                             batch=batch,
                                             weight_mask = batch["weights"],
                                             possible_mask = possible_logit)
                
            for metric, value in validation_state.metrics.compute().items():
                metric_history[f'validation_{metric}'].append(value)
            # Log messages
            print('Epoch {} train/val loss: {:.6f} / {:.6f}'.format(epoch,
                                                                    metric_history['train_loss'][-1],
                                                                    metric_history['validation_loss'][-1]))
            
            validation_loss = metric_history['validation_loss'][-1]
            if validation_loss < min_validation_loss:
                min_validation_loss = validation_loss
                ckpt = {'model': state, 'epoch':epoch, 
                        'metric_history': metric_history, 'min_loss': min_validation_loss}
                self._save_checkpoint(ckpt, 'best')
            if epoch % self.config.save_every_epoch == 0:
                ckpt = {'model': state, 'epoch':epoch, 
                        'metric_history': metric_history, 'min_loss': min_validation_loss}
                self._save_checkpoint(ckpt, f'checkpoint_epoch{epoch}')
    
    def test(self):
        #Setup Dataset and DataLoader
        test_dataset = LayoutDataset('test',
                                      self.config.dataset_path+'/test.json',
                                      config=self.config.dataset)
        collator = PaddingCollator(pad_token_id=test_dataset.pad_idx, seq_len=test_dataset.seq_len)
        test_dataloader = DataLoader(test_dataset, batch_size=self.config.eval_batch_size,
                                    collate_fn=collator.collate_padding,
                                    shuffle=self.config.train_shuffle)
        # Setup model's state
        init_batch = jnp.ones((self.config.batch_size, test_dataset.seq_len))
        init_label = jnp.ones((self.config.batch_size, 1))
        init_batch = dict(inputs=init_batch, labels=init_label)
        state =  self.create_train_state(rng = self.rng, inputs = init_batch)

        if self.config.checkpoint_path is not None:
            target = {'model':state, 'metric_history':dict, 'min_loss': float, 'epoch':int}
            ckpt = self._load_checkpoint(target)
            state = ckpt['model']

            metric_history = ckpt['metric_history']
            min_validation_loss = ckpt['min_loss']
            start_epoch = ckpt['epoch']

            print("Loaded epoch {} - Loss: {:.6f}".format(start_epoch, min_validation_loss))
        else:
            raise TypeError("Please provide a checkpoint path!!")
        
        # Get possible logit for each position
        vocab_size = test_dataset.get_vocab_size()
        pos_info = [[test_dataset.offset_class, test_dataset.number_classes], 
                    [test_dataset.offset_width, test_dataset.resolution_w],
                    [test_dataset.offset_height, test_dataset.resolution_h],
                    [test_dataset.offset_center_x, test_dataset.resolution_w],
                    [test_dataset.offset_center_y, test_dataset.resolution_h]]
        seq_len = test_dataset.seq_len
        possible_logit, _ = self._make_possible_mask(vocab_size=vocab_size, pos_info=pos_info,seq_len=seq_len)
        
    
    @partial(jit, static_argnums=(0,))
    def train_step(self,
                   state,
                   batch,
                   possible_mask = None):
        def loss_fn(params):
            logits = state.apply_fn({'params':params}, input_ids=batch["masked_inputs"], labels=None)
            #loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            loss = self._compute_weighted_cross_entropy(logits, batch["targets"], batch["weights"], possible_mask)
            return loss
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state
    
    @partial(jit, static_argnums=(0,))
    def compute_metrics(self, 
                        state, 
                        batch, 
                        possible_mask = None):
        logits = state.apply_fn({'params': state.params}, input_ids=batch['masked_inputs'], labels=None)
        loss = self._compute_weighted_cross_entropy(logits, batch["targets"], batch["weights"], possible_mask)
        metric_updates = state.metrics.single_from_model_output(loss = loss)
        metrics = state.metrics.merge(metric_updates)
        state = state.replace(metrics=metrics)
        return state
    
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
            if idx == 0:
                logit_mask = logit_mask.at[:,:,2].set(0)
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
        #mask out impossible tokens
        if logit_mask is not None:
            logits = jnp.where(logit_mask>0, 1e-7, logits)

        vocab_size = logits.shape[-1]
        confidence = 1 - label_smoothing
        low_confidence = label_smoothing / (vocab_size-1)
        soft_targets = common_utils.onehot(targets, vocab_size,
                                           on_value=1.0, off_value=low_confidence)
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
    
    def _create_optimizer(self):
        opt_def = optax.adam(
            learning_rate=self.config.optimizer.lr,
            b1=self.config.optimizer.beta1,
            b2=self.config.optimizer.beta2,
        )
        return opt_def
    
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
