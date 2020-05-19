# -*- coding: utf-8 -*-
from csv import reader
from random import randrange

# the best performing split point is selected
def best_split(ds, n_attrib):
    # run over the possible values that the class variable entails
	values_cls = list(set(row[-1] for row in ds))
	res_index, res_value, res_score, res_groups = 1111, 1111, 1111, None
	attrib = list()
	while len(attrib) < n_attrib:
		index = randrange(len(ds[0])-1)
		if index not in attrib:
			attrib.append(index)
	for index in attrib:
		for row in ds:
			ds_groups = split_ds(index, row[index], ds)
			gini_ind = gini_idx(ds_groups, values_cls)
			if gini_ind < res_score:
				res_index, res_value, res_score, res_groups = index, row[index], gini_ind, ds_groups
	return {'index':res_index, 'value':res_value, 'groups':res_groups}

# Gini index is then calculated for the partitioned ds
def gini_idx(groups, classes):
	# number of samples at the pont of split is obtained below
	n_samples = float(sum([len(group) for group in groups]))
	# the iterations below calculate the sum of weighted Gini indices for each of the groups
	n_gini = 0.0
	for group in groups:
		size = float(len(group))
		# check
		if size == 0:
			continue
		score = 0.0
		# the group is scored based on calculated proportion for each class per group
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# the group score is weighted accordingly
		n_gini += (1.0 - score) * (size / n_samples)
	return n_gini

# this method splits the ds based on factor and factor value
def split_ds(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
#implement main()