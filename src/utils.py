import jax
import jax.numpy as jnp
import ml_collections

import functools

def attribute_random_masking(inputs, mask_token, pad_token, layout_dim):
    targets = inputs
    total_dim = layout_dim*2 + 1
    rng = jax.random.PRNGKey(jnp.sum(inputs, dtype='int32'))

    # Get positions (in the tokenized sequence)
    # - [PAD] token position
    # - element's class position
    # - element's width height position
    # - element's center position
    is_pad = inputs == pad_token
    position_ids = jnp.arange(inputs.shape[-1])[None, :]
    is_asset = position_ids % total_dim == 0
    is_size = functools.reduce(
        lambda x,y: x|y,
        [position_ids % total_dim == i for i in range(1, layout_dim+1)]
    )
    is_position = functools.reduce(
        lambda x,y: x | y,
        [position_ids % total_dim == i for i in range(layout_dim+1, total_dim)]
    )

    
    rand = jax.random.uniform(rng, (inputs.shape[0],1))
    pre_masked_inputs = inputs.copy()
    # all token are [MASKED]
    fullmask = jnp.full_like(inputs, mask_token)
    fullmask = jnp.where(is_pad, pad_token, fullmask) # [PAD] token

    weights = is_asset & (~is_pad)
    weights = jnp.where(
        jnp.logical_and(rand>=0.2, rand<0.4), is_size & (~is_pad), weights)
    weights = jnp.where(
        jnp.logical_and(rand>=0.4, rand<0.6), is_position & (~is_pad), weights)
    weights = jnp.where(
        jnp.logical_and(rand>=0.6, rand<0.8), is_size & (~is_pad), weights)
    weights = jnp.where(rand>0.8, is_asset & ~(is_pad), weights)
    
    lens = jnp.sum(weights, axis=-1)
    rng, subrng = jax.random.split(rng)
    mask_rate = 1 - jax.random.uniform(subrng, lens.shape)

    mask_lens = jax.lax.ceil(lens*mask_rate).astype('int32')
    rng, subrng = jax.random.split(rng)
    
    should_mask = jax.random.uniform(subrng, shape=inputs.shape)
    should_mask = jnp.where(weights, should_mask, 2.)
    sorted_should_mask = jnp.sort(should_mask, axis=-1)
    cut_off = jnp.take_along_axis(
        sorted_should_mask, jnp.expand_dims(mask_lens-1, 1), axis=-1)
    should_mask = jnp.where(should_mask <= cut_off, 1, 0)
    fullmask = jnp.full_like(inputs, mask_token)

    masked_inputs = jnp.where(should_mask, fullmask, pre_masked_inputs)
    weights = jnp.where(is_pad, 0, should_mask)
    return dict(masked_inputs=masked_inputs, targets=targets, weights=weights)

def get_magazine_config():
  """Gets the default hyperparameter configuration."""

  config = ml_collections.ConfigDict()
  # Exp info
  config.dataset_path = "/path/to/magazine"
  config.dataset = "MAGAZINE"
  config.vocab_size = 137
  config.experiment = "bert_layout"
  config.model_class = "bert_layout"
  config.image_size = 256

  # Training info
  config.layout_dim = 2
  config.seed = 0
  config.log_every_steps = 100
  config.eval_num_steps = 1000
  config.max_length = 130
  config.batch_size = 64
  config.train_shuffle = True
  config.eval_pad_last_batch = False
  config.eval_batch_size = 64
  config.num_train_steps = 30_000
  config.checkpoint_every_steps = 5000
  config.eval_every_steps = 5000
  config.num_eval_steps = 100

  # Model info
  config.dtype = "float32"
  config.autoregressive = False
  config.shuffle_buffer_size = 10
  config.use_vae = True
  config.share_embeddings = True
  config.num_layers = 4
  config.qkv_dim = 512
  config.emb_dim = 512
  config.mlp_dim = 2048
  config.num_heads = 8
  config.dropout_rate = 0.1
  config.attention_dropout_rate = 0.3
  config.restore_checkpoints = True
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

  return config