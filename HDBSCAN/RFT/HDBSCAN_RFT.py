import pandas as pd
import numpy as np
import time
from sklearnex import patch_sklearn
patch_sklearn()

import fast_hdbscan
from sklearn.kernel_approximation import RBFSampler
import csv

import os

print("Packages Loaded!!!")

vector_path = os.getcwd() + "/../../feature_frequency_vector.npy"
variant_path = os.getcwd() + "/../../filtered_variants.npy"

X = np.load(vector_path)
y_orig = np.load(variant_path)

print("Attributed data Reading Done")

unique_varaints = list(np.unique(y_orig))

int_variants = []
for ind_unique in range(len(y_orig)):
    variant_tmp = y_orig[ind_unique]
    ind_tmp = unique_varaints.index(variant_tmp)
    int_variants.append(ind_tmp)

print("Attribute data preprocessing Done")

y =  np.array(int_variants)

from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.9)
train_index, test_index = next(sss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
y_train_orig, y_test_orig = y_orig[train_index], y_orig[test_index]

print("Train-Test Split Done")

print("X_train rows = ",X_train.shape[0],"X_train columns = ",X_train.shape[1])
print("X_test rows = ",X_test.shape[0],"X_test columns = ",X_test.shape[1])

print("Lasso Features Starts here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

start_time_ = time.time()

rbf_feature = RBFSampler(gamma=1, n_components=500)
X_features_train = rbf_feature.fit_transform(X_train)
X_features_test = rbf_feature.transform(X)

end_time_ = time.time() - start_time_
print("Feature Dimension Reduction in seconds =>", end_time_)
######################################################################################

start_time = time.time()

number_of_clusters = 5 #number of clusters

print("Number of Clusters = ",number_of_clusters)
clust_num = number_of_clusters

clusterer = fast_hdbscan.HDBSCAN(min_cluster_size=clust_num)
HDBSCAN_labels = clusterer.fit_predict(X_features_test)

np.save(os.getcwd() + '/new_Labels_HDBSCAN_RFT.npy', HDBSCAN_labels)

end_time = time.time() - start_time
print("Clustering Time in seconds =>",end_time)

write_path_112 = os.getcwd() + "/new_int_true_variants_k_" + str(clust_num) + ".csv"

pd.DataFrame(y).to_csv(write_path_112, header=False, index=False)

write_path_11 = os.getcwd() + "new_orig_true_variants_k_" + str(clust_num) + ".csv"

pd.DataFrame(y_orig).to_csv(write_path_11, header=False, index=False)

print("All Processing Done!!!")
