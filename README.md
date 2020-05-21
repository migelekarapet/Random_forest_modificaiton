###Random_forest_modificaiton
###implementation of Random forest modification
 step 1. randomly select observations from dataset (note repeated observations are allowed)
 step 2. randomly select a subset of variables(columns) at each step; and create a decision tree using bootstraped dataset from step 1
 we are considering only fixed limited number of variables at each step. an explanation will follow later on how to determine 
 optimal  of variables
 now, say we have 4 column variables and our fixed number is set to 2; we select 2 random columns and see which ones are best 
 candidates for root node. assume one of them did the best job separating samples; we logically grey out this variable column
 and then focus on the remaining 3 variable columns; note - the root node is already selected and imagine for simplicity
 thatwe have ledt and right child nodes non empty. Now say we are at left child node. we then randomly select 
  subset (2 out of 3 columns-variables) and we just build the tree as usual but only considering random subset of variables at each substep
 we built a tree using a) bootstraped dataset and b) only considering a random subset of variables at each step

 Now, we repeat again the process from step 1 onwards - that is, we make a new bootstraped ds and build a tree considering
 a subset of variables at each step; ideally it's done numerous times, depending on size of ds
 this results in a wide variety of trees. Thsi variety is what's making the random forest more efficient than the individual decision tree alon

 now when we've got the random forset - how do we utilize it?
 suppose a new observation arrives. We take it and run it down the first tree that we've made. say we obtained some classification 
(i.e. ended up at seartain leaf node at very bottom of the tree). we keep track of that answer. Then we run it==the observaiton through
 the second tree and we again keep track of that answer, etc. etc. After executing the data for all the trees in random forest we see 
which option received more votes. And for that particular observation that vote is considered the answer. Same is done for subsequent obs.
 bootstrapping and using aggregate answers to calculate final vote is called bagging altogether
 note: normally around 30 % of data is kept away from the bootstraped ds (out-of-bag subset)
 on each step of creation a new tree for that particular tree we can check if the out-of-bag subset (unseen data) performs correctly
 running an out-of-bag subset over the trees that were built w/o it. We calculate the votes for that  out-of-bag subset to see how correctly
 the random forest classified out-of-bag subset.We then do the same for all out-of-bag subsets for all of trees that were built w/o 'em
 recordings of results show how many times the out-of-bag subsets were correctly labeled (per tree) and how many times - they were not.
 etc., etc., etc. By calculating the proportion of out-of-bag subsets that were correctly classified by random forest we can combine
 those frequencies and derive conclusions as to how correct our RF does its job (out-of-bag error is the proportion of inaccurately classified)
