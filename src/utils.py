import jax
import jax.numpy as jnp
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
		jnp.logical_and(rand >= 0.33, rand < 0.67), is_size & (~is_pad), weights)
	weights = jnp.where(rand >= 0.67, is_position & (~is_pad), weights)
	
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
	config.save_every_epoch = 25

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
	config.save_every_epoch = 25

	# Model info
	config.layout_dim = 2
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
	config.checkpoint_path = None
	config.dataset_path = "data/obello/all"
	config.experiment = "bert_layout"
	config.model_class = "bert_layout"
	config.image_size = 256

	# Training info
	config.epoch = 350
	config.seed = 0
	config.max_length = 130
	config.batch_size = 64
	config.train_shuffle = True
	config.eval_pad_last_batch = False
	config.eval_batch_size = 64
	config.save_every_epoch = 25

	# Model info
	config.layout_dim = 2
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
	config.dataset.LABEL_NAMES = ('Headline', 'Body', 'Logo', 'Image', 'CTA', 'button', 'text_in_shape', 'shape')
	config.dataset.COLORS = {
			'Headline' : (193, 0, 0),
			'Body' : (0, 193, 0),
			'Logo' : (0, 0, 193),
			'Image': (128, 128, 0),
			'CTA': (0, 128, 128),
			'button': (128, 0, 128),
			'text_in_shape' : (64, 128, 0),
			'shape' : (128, 0, 64),
			'background': (200,200,200)
			}

	config.dataset.FRAME_WIDTH = 1050
	config.dataset.FRAME_HEIGHT = 1485

	config.dataset.ID_TO_LABEL = frozendict.frozendict({
			0:'Headline',
			1:'Body',
			2:'Logo',
			3:'Image',
			4:'CTA',
			5:'button',
			6:'text_in_shape',
			7:'shape',
		})
	config.dataset.NUMBER_LABELS = 8
	config.dataset.LABEL_TO_ID_ = frozendict.frozendict(
			{l: i for i,l in config.dataset.ID_TO_LABEL.items()})
	return config