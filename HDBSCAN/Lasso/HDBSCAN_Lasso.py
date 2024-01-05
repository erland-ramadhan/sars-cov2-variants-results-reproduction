import pandas as pd
import numpy as np
import time
from sklearnex import patch_sklearn, config_context
patch_sklearn()
from joblib import Memory

# import hdbscan
import fast_hdbscan
import csv

import os
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

print("Packages Loaded!!!")

vector_path = os.getcwd() + '/../../feature_frequency_vector.npy'
variant_path = os.getcwd() + '/../../filtered_variants.npy'

frequency_vector_read_final = np.load(vector_path)
variant_orig = np.load(variant_path)

print("Attributed data Reading Done")

unique_varaints = list(np.unique(variant_orig))

int_variants = []
for ind_unique in range(len(variant_orig)):
    variant_tmp = variant_orig[ind_unique]
    ind_tmp = unique_varaints.index(variant_tmp)
    int_variants.append(ind_tmp)

print("Attribute data preprocessing Done")

X = np.array(frequency_vector_read_final)
y =  np.array(int_variants)
y_orig = np.array(variant_orig)

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

scaler = StandardScaler()

# L1 = Lasso, L2 = Ridge
sel_ = SelectFromModel(estimator=LogisticRegression(n_jobs=-1, penalty='l1', solver='liblinear'))
sel_.fit(scaler.fit_transform(X_train), y_train)

print('total features: {}'.format(X_train.shape[1]))
print('selected features: {}'.format(np.sum(sel_.get_support() == True)))
print('features with coefficients shrank to zero: {}'.format(
      np.sum(sel_.estimator_.coef_ == 0)))

X_features_test = sel_.transform(X)

end_time_ = time.time() - start_time_
print("Feature Dimension Reduction Time in seconds =>", end_time_)
######################################################################################
start_time = time.time()

number_of_clusters = 5 #number of clusters

print("Number of Clusters = ",number_of_clusters)
clust_num = number_of_clusters

clusterer = fast_hdbscan.HDBSCAN(min_cluster_size=clust_num)
HDBSCAN_labels = clusterer.fit_predict(X_features_test)

np.save(os.getcwd() + '/new_Labels_HDBSCAN_Lasso.npy', HDBSCAN_labels)

end_time = time.time() - start_time
print("Clustering Time in seconds =>",end_time)

write_path_112 = os.getcwd() + '/new_int_true_variants_k_' + str(clust_num) + ".csv"

pd.DataFrame(y).to_csv(write_path_112, header=False, index=False)

write_path_11 = os.getcwd() + "new_orig_true_variants_k_" + str(clust_num) + ".csv"

pd.DataFrame(y_orig).to_csv(write_path_11, header=False, index=False)

print("All Processing Done!!!")

