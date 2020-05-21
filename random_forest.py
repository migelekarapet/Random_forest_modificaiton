# -*- coding: utf-8 -*-
from random import seed
from random import randrange
from csv import reader
from math import sqrt

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

# Gini index is then calculated for the partitioned our ds
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
# step 1. randomly select observations from dataset (note repeated observations are allowed)
# step 2. randomly select a subset of variables(columns) at each step; and create a decision tree using bootstraped dataset from step 1
# we are considering only fixed limited number of variables at each step. an explanation will follow later on how to determine 
# optimal # of variables
# now, say we have 4 column variables and our fixed number is set to 2 
#(number of attributes to be considered for the split is limited to the square root of the number of input features); 
# as a result  of this approachwe obtain uncorrelated trees and the final  predictions are more diverse 
# Hence, we get a combined prediction most of times has better performance as opposed to  single tree (or bagging).
#we select 2 random columns and see which ones are best 
# candidates for root node. assume one of them did the best job separating samples; we logically grey out this variable column
# and then focus on the remaining 3 variable columns; note - the root node is already selected and imagine for simplicity
# thatwe have ledt and right child nodes non empty. Now say we are at left child node. we then randomly select 
#  subset (2 out of 3 columns-variables) and we just build the tree as usual but only considering random subset of variables at each substep
# we built a tree using a) bootstraped dataset and b) only considering a random subset of variables at each step

# Now, we repeat again the process from step 1 onwards - that is, we make a new bootstraped ds and build a tree considering
# a subset of variables at each step; ideally it's done numerous times, depending on size of ds
# this results in a wide variety of trees. Thsi variety is what's making the random forest more efficient than the individual decision tree alon

# now when we've got the random forset - how do we utilize it?
# suppose a new observation arrives. We take it and run it down the first tree that we've made. say we obtained some classification 
#(i.e. ended up at seartain leaf node at very bottom of the tree). we keep track of that answer. Then we run it==the observaiton through
# the second tree and we again keep track of that answer, etc. etc. After executing the data for all the trees in random forest we see 
#which option received more votes. And for that particular observation that vote is considered the answer. Same is done for subsequent obs.
# bootstrapping and using aggregate answers to calculate final vote is called bagging altogether
# note: normally around 30 % of data is kept away from the bootstraped ds (out-of-bag subset)
# on each step of creation a new tree for that particular tree we can check if the out-of-bag subset (unseen data) performs correctly
# running an out-of-bag subset over the trees that were built w/o it. We calculate the votes for that  out-of-bag subset to see how correctly
# the random forest classified out-of-bag subset.We then do the same for all out-of-bag subsets for all of trees that were built w/o 'em
# recordings of results show how many times the out-of-bag subsets were correctly labeled (per tree) and how many times - they were not.
# etc., etc., etc. By calculating the proportion of out-of-bag subsets that were correctly classified by random forest we can combine
# those frequencies and derive conclusions as to how correct our RF does its job (out-of-bag error is the proportion of inaccurately classified)


# generate random subset from ds (with repl.)
def subsets(ds, prop):
	observ = list()
	n_observ = round(len(ds) * prop)
	while len(observ) < n_observ:
		index = randrange(len(ds))
		observ.append(ds[index])
	return observ

# a conversion of string column to float is performed below
def str_column_to_float(ds, column):
	for row in ds:
		row[column] = float(row[column].strip())
 
# a conversion of string column to int is performed below
def str_column_to_int(ds, column):
	values_cls = [row[column] for row in ds]
	unique = set(values_cls)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in ds:
		row[column] = lookup[row[column]]
	return lookup
 
# ds is split k times (i.e. into k folds)
def split_cval(ds, n_times):
    ds_copy = list(ds)
    ds_split = list()
    times_size = int(len(ds) / n_times)
    for i in range(n_times):
        times = list()
        while len(times) < times_size:
            ind = randrange(len(ds_copy))
            times.append(ds_copy.pop(ind))
            ds_split.append(times)
    return ds_split
 
# acc %
def acc_metric(act, predic):
    #counter is set to zero first time
	correct = 0
	for i in range(len(act)):
		if act[i] == predic[i]:
			correct += 1
    #the right proportion is returned
	return correct / float(len(act)) * 100.0
 
# we use k-times cross validation split below
def eval(ds, algo, n_times, *args):
	timess = split_cval(ds, n_times)
	scr = list()
	for times in timess:
		train_set = list(timess)
		train_set.remove(times)
		train_set = sum(train_set, [])
		test_set = list()
		for rw in times:
			row_copy = list(rw)
			test_set.append(row_copy)
			row_copy[-1] = None
		m_predic = algo(train_set, test_set, *args)
		m_act = [rw[-1] for rw in times]
		acc = acc_metric(m_act, m_predic)
		scr.append(acc)
	return scr


def csv_load(filename):
	ds = list()
	with open(filename, 'r') as file:
		csv_rd = reader(file)
		for row in csv_rd:
			if not row:
				continue
			ds.append(row)
	return ds
# terminating node value creaiton
def terminating(group):
	results = [row[-1] for row in group]
	return max(set(results), key=results.count)

# make terminating or creation of node's child split
def split_node(node, depth_max, size_min, n_attrib,  depth):
	left, right = node['groups']
	del(node['groups'])
	# emptyness check
	if not left or not right:
		node['left'] = node['right'] = terminating(left + right)
		return
	# verifying if we reached max depth
	if depth >= depth_max:
		node['left'], node['right'] = terminating(left), terminating(right)
		return
	# node's left child
	if len(left) <= size_min:
		node['left'] = terminating(left)
	else:
		node['left'] = best_split(left)
		split_node(node['left'], depth_max, size_min, n_attrib, depth+1)
	# node's right child
	if len(right) <= size_min:
		node['right'] = terminating(right)
	else:
		node['right'] = best_split(right)
		split_node(node['right'], depth_max, size_min, n_attrib, depth+1)
 


# decision tree construciton
def dec_tree_construct(trn, depth_max, size_min, n_attrib):
	root = best_split(trn, n_attrib)
	split_node(root, depth_max, size_min, n_attrib, 1)
	return root

#  we'll now perform a prediction for decision tree's most likely node 
def node_prediction(nd, row):
	if row[nd['index']] < nd['value']:
		if isinstance(nd['left'], dict):
			return node_prediction(nd['left'], row)
		else:
			return nd['left']
	else:
		if isinstance(nd['right'], dict):
			return node_prediction(nd['right'], row)
		else:
			return nd['right']

# Using list of baggd treed we'll make a predict
def pred_bagg(trs, rw):
	predic = [node_prediction(tr, rw) for tr in trs]
	return max(set(predic), key=predic.count)

# RF
def r_f(trn, tst, depth_max, size_min, subset_len, n_trs, n_attrib):
	trs = list()
	for i in range(n_trs):
		subset = subsets(trn, subset_len)
		tr = dec_tree_construct(subset, depth_max, size_min, n_attrib)
		trs.append(tr)
	pred = [pred_bagg(trs, rw) for rw in tst]
	return(pred)
 


# RF building
seed(71)
# loading the data
filename = 'default of credit card clients.csv'
ds = csv_load(filename)


for i in range(0, len(ds[0])-1):
	str_column_to_float(ds, i)
# convert class column to integers
str_column_to_int(ds, len(ds[0])-1)
# evalalgorithm
n_folds = 5
depth_max = 10
size_min = 1
sample_size = 1.0
n_attrib = int(sqrt(len(ds[0])-1))
for n_trees in [1, 5, 10]:
	scores = eval(ds, r_f, n_folds, depth_max, size_min, sample_size, n_trees, n_attrib)
	print('Trees: %d' % n_trees)
	print('Scores: %s' % scores)
	print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
