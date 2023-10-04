import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
import frozendict

import functools

def normalize_bbox(layout,
				   resolution_w = 32,
				   resolution_h = 32):
	"""Normalize the bounding box.
	
	Args
	----
	layout : np.array
		An iterable of normalize bounding box coordinate
		in the format of (class, width, height, center_x, center_y)
	resolution_w : int
		the width of the model input
	resolution_h : int
		the height of the model input
	
	Return
	------
	layout : np.array
		An iterable of normalize bounding box coordinate 
		in the format of (class, x_min, y_min, x_max, y_max)
	"""
	layout = np.array(layout, dtype=np.float32)
	layout = np.reshape(layout, (-1, 5))
	width, height = np.copy(layout[:,1]), np.copy(layout[:, 2])
	layout[:, 1] = (layout[:, 3] - width / 2.) / resolution_w
	layout[:, 2] = (layout[:, 4] - height / 2.) / resolution_h
	layout[:, 3] = (layout[:, 3] + width / 2.) / resolution_w
	layout[:, 4] = (layout[:, 4] + height / 2.) / resolution_h
	return layout[:, 1:]

def get_layout_iou(layout):
	"""Computes the IOU on the layout level.
	
	Args
	----
	layout : np.array
		1-d integer array in which every 5 elements form
		an layout eleemnt in the format (class, width, height, center_x, center_y)
	
	Return
	------
	The value for the overlap index. 0 is return if no overlap are found.
	"""
	layout = np.array(layout, dtype=np.float32)
	layout = np.reshape(layout, (-1,5))
	layout_channels = []
	for bbox in layout:
		canvas = np.zeros((32, 32, 1), dtype=np.float32)
		width, height = bbox[1], bbox[2]
		center_x, center_y = bbox[3], bbox[4]
		min_x = round(center_x - width/2. + 1e-4)
		max_x = round(center_x + width/2. + 1e-4)
		min_y = round(center_y + height/2. + 1e-4)
		max_y = round(center_y + height/2. + 1e-4)
		canvas[min_x:max_x, min_y:max_y] = 1
		layout_channels.append(canvas)
	if not layout_channels:
		return 0
	sum_layout_channel = np.sum(np.concatenate(layout_channels, axis=-1), axis=-1)
	overlap_area = np.sum(np.greater(sum_layout_channel, 1.))
	bbox_area = np.sum(np.greater(sum_layout_channel, 0.))
	if bbox_area == 0:
		return 0
	return overlap_area/bbox_area

def get_average_iou(layout):
	iou_values = []
	layout = normalize_bbox(layout)
	for i in range(len(layout)):
		for j in range(i+1, len(layout)):
			bbox1 = layout[i]
			bbox2 = layout[j]
			iou_for_pair = get_iou(bbox1, bbox2)
			if iou_for_pair > 0:
				iou_values.append(iou_for_pair)
	return np.mean(iou_values) if len(iou_values) else 0.

def get_iou(bb0, bb1):
	intersection = get_intersection(bb0, bb1)
	bb0_area = area(bb0)
	bb1_area = area(bb1)
	if np.isclose(bb0_area + bb1_area - intersection, 0.):
		return 0
	return intersection / (bb0_area + bb1_area - intersection)

def get_intersection(bb0, bb1):
	x_0, y_0, x_1, y_1 = bb0
	u_0, v_0, u_1, v_1 = bb1
	intersection_x_0 = max(x_0, u_0)
	intersection_y_0 = max(y_0, v_0)
	intersection_x_1 = min(x_1, u_1)
	intersection_y_1 = min(y_1, v_1)
	intersection = area(
		[intersection_x_0, intersection_y_0, intersection_x_1, intersection_y_1]
	)
	return intersection

def area(bbox):
	x_0, y_0, x_1, y_1 = bbox
	return max(0.,x_1 - x_0) * max(0., y_1 - y_0)

def get_overlap_index(layout):
	intersection_areas = []
	layout = normalize_bbox(layout)
	for i in range(len(layout)):
		for j in range(i+1, len(layout)):
			bbox1 = layout[i]
			bbox2 = layout[j]

			intersection = get_intersection(bbox1, bbox2)
			if intersection > 0.:
				intersection_areas.append(intersection)
	return np.sum(intersection_areas) if intersection_areas else 0.

def get_alignment_loss(layout):
	layout = normalize_bbox(layout)
	if len(layout) <= 1:
		return -1
	return get_alignment_loss_numpy(layout)

def get_alignment_loss_numpy(layout):
	a = layout
	b = layout
	a, b = a[None,:, None], b[:, None, None]
	cartersian_product = np.concatenate(
		[a + np.zeros_like(b), np.zeros_like(a) + b], axis=2)

	left_correlation = left_similarity(cartersian_product)
	center_correlation = center_similarity(cartersian_product)
	right_correlation = right_similarity(cartersian_product)
	correlations = np.stack(
		[left_correlation, center_correlation, right_correlation], axis=2)
	min_correlation = np.sum(np.min(correlations, axis = (1,2)))
	return min_correlation

def left_similarity(correlated):
	remove_diagonal_entries_mask = np.zeros(
		(correlated.shape[0], correlated.shape[0]))
	np.fill_diagonal(remove_diagonal_entries_mask, np.inf)
	correlations = np.mean(
		np.abs(correlated[:, :, 0, :2] -  correlated[:, : , 1, :2]), axis=-1)
	return correlations + remove_diagonal_entries_mask

def right_similarity(correlated):
	remove_diagonal_entries_mask = np.zeros(
		(correlated.shape[0], correlated.shape[0]))
	np.fill_diagonal(remove_diagonal_entries_mask, np.inf)
	correlations = np.mean(
		np.abs(correlated[:, :, 0, 2:] - correlated[:, :, 1, 2:]), axis=-1)
	return correlations + remove_diagonal_entries_mask

def center_similarity(correlated):
	x0 = (correlated[:,:,0,0] + correlated[:,:,0,2])/2
	y0 = (correlated[:,:,0,1] + correlated[:,:,0,3])/2
	centroids0 = np.stack([x0,y0], axis=2)

	x1 = (correlated[:,:,1,0] + correlated[:,:,1,2])/2
	y1 = (correlated[:,:,1,1] + correlated[:,:,1,3])/2
	centroids1 = np.stack([x1,y1], axis=2)

	correlations = np.mean(np.abs(centroids0-centroids1), axis=-1)
	remove_diagonal_entries_mask = np.zeros(
		(correlated.shape[0], correlated.shape[0]))
	np.fill_diagonal(remove_diagonal_entries_mask, np.inf)
	
	return correlations + remove_diagonal_entries_mask

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
	config.vocab_size = 137
	config.experiment = "bert_layout"
	config.model_class = "bert_layout"
	config.image_size = 256

	# Training info
	config.epoch = 5000
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