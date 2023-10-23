import numpy as np
import functools
import scipy

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

def docsim_w(b1, b2, conditional='none'):
	if b1[0] != b2[0]:
		return 0.
	if conditional == 'a+s':
		if b1[1] != b2[1] or b1[2] != b2[2]:
			return 0.
	alpha = min(b1[5], b2[5]) ** 0.5

	delta_c = np.linalg.norm(b1[3:5] - b2[3:5])
	delter_s = np.sum(np.abs(b1[1:3] - b2[1:3]))

	w = alpha * 2**(-delta_c - 2 * delter_s)
	return w

def docsim(layout_1, layout_2,
		   max_element_count_diff=1, conditional='none'):
	n_asset_1 = layout_1.shape[0]
	n_asset_2 = layout_2.shape[0]

	if n_asset_1 == 0 or n_asset_2 == 0:
		return 0.
	distance_function = functools.partial(docsim_w, 
									   conditional=conditional)
	if max_element_count_diff is not None and abs(
		n_asset_1 - n_asset_2) > max_element_count_diff:
		return 0.
	
	distances = scipy.spatial.distance.cdist(
		layout_1, layout_2, metric=distance_function)
	assignment_row, assignment_col = scipy.optimize.linear_sum_assignment(
		-distances)

	costs = distances[assignment_row, assignment_col]

	result = 1 / costs.size * costs.sum()
	return result

def preprocess_layouts_for_distance(layouts_lst):
	normalized_layouts = []
	for layouts in layouts_lst:
		layouts = np.array(layouts, dtype=np.float32)
		layouts[:,1:] = layouts[:, 1:] /32
		areas = layouts[:, 1] * layouts[:, 2]
		layouts = np.concatenate((layouts, areas[:,None]), axis=1)
		normalized_layouts.append(layouts)
	return normalized_layouts

def conditional_distance(synthetic_layouts,
						 real_layouts,
						 conditional='a'):
	synthetic_layouts = preprocess_layouts_for_distance(synthetic_layouts)
	real_layouts = preprocess_layouts_for_distance(real_layouts)

	distances = []
	for s_layout, r_layout in zip(synthetic_layouts, real_layouts):
		docsim_distance = docsim(s_layout, r_layout, conditional=conditional)
		distances.append(docsim_distance)
	
	distances = np.asarray(distances)
	return np.mean(distances)

def diversity(synthetics_layouts):
	synthetics_layouts = preprocess_layouts_for_distance(synthetics_layouts)
	distances = []
	for i in range(len(synthetics_layouts)):
		for j in range(i+1, len(synthetics_layouts)):
			docsim_distance = docsim(synthetics_layouts[i],
									synthetics_layouts[j])
			distances.append(docsim_distance)
	return sum(distances) / len(distances)

def unique_match(synthetic_layouts,
				 real_layouts):
	synthetic_layouts = preprocess_layouts_for_distance(synthetic_layouts)
	real_layouts = preprocess_layouts_for_distance(real_layouts)

	distances = []
	for s_layout in synthetic_layouts:
		dist_for_layout = []
		for r_layout in real_layouts:
			docsim_distance = docsim(
				s_layout,
				r_layout)
			dist_for_layout.append(docsim_distance)
		distances.append(dist_for_layout)
	distances = np.asarray(distances)
	matching_indices = np.argmax(distances, axis=-1)
	unique_indices = np.unique(matching_indices)

	return unique_indices.size