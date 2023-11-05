import pandas as pd
import numpy as np
import time
from sklearnex import patch_sklearn # config_context
patch_sklearn()

# from sklearn.svm import SVC
#import RandomBinningFeatures
# from sklearn.kernel_approximation import RBFSampler
# from sklearn.linear_model import RidgeClassifier
# from sklearn import metrics
# import scipy
# import matplotlib.pyplot as plt
#%matplotlib inline
import csv

import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from sklearn.decomposition import TruncatedSVD
# import random
# import seaborn as sns
# import os.path as path
import os
# import matplotlib
# import matplotlib.font_manager
# import matplotlib.pyplot as plt # graphs plotting
# import Bio
# from Bio import SeqIO # some BioPython that will come in handy
#matplotlib inline

# from itertools import cycle

# from sklearn import svm, datasets
# from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import label_binarize
# from sklearn.multiclass import OneVsRestClassifier
# from scipy import interp
# from sklearn.metrics import roc_auc_score


# from matplotlib import rc
# # for Arial typefont
# matplotlib.rcParams['font.family'] = 'Arial'

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Lasso, LogisticRegression
# from sklearn.feature_selection import SelectFromModel
# from sklearn.preprocessing import StandardScaler

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn import metrics
# from sklearn import svm

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn import metrics

# import sklearn
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import classification_report, confusion_matrix

# from sklearn.neighbors import KNeighborsClassifier

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix


# from sklearn.model_selection import KFold
# from sklearn.model_selection import RepeatedKFold

# from sklearn.metrics import confusion_matrix

# from sklearn.linear_model import LogisticRegression
#from boruta import BorutaPy
#from sklearn.ensemble import RandomForestClassifier
#import seaborn as sns
# from boruta import BorutaPy
# from sklearn.ensemble import RandomForestClassifier




print("Packages Loaded!!!")


# # Read Frequency Vector

vector_path = os.getcwd() + '/../../feature_frequency_vector.npy'
variant_path = os.getcwd() + '/../../filtered_variants.npy'


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

# X = np.array(frequency_vector_read_final)
y =  np.array(int_variants, dtype=np.uint8)
# y_orig = np.array(variant_orig)

# from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit
# sss = ShuffleSplit(n_splits=1, test_size=0.9)
# sss.get_n_splits(X, y)
# train_index, test_index = next(sss.split(X, y))

# X_train, X_test = X[train_index], X[test_index]
# y_train, y_test = y[train_index], y[test_index]
# y_train_orig, y_test_orig = y_orig[train_index], y_orig[test_index]

# print("Train-Test Split Done")


# print("X_train rows = ",len(X_train),"X_train columns = ",len(X_train[0]))
# print("X_test rows = ",len(X_test),"X_test columns = ",len(X_test[0]))

# print("Lasso Features Starts here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# start_time = time.time()

# rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
# # define Boruta feature selection method
# feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=1)
# # find all relevant features - 5 features should be selected
# feat_selector.fit(X_train, y_train)
# # call transform() on X to filter it down to selected features
# X_features_test = feat_selector.transform(X)
# y =  np.array(y)
######################################################################################

start_time = time.time()

#for clustering, the input data is in variable X_features_test
from sklearn.cluster import KMeans


number_of_clusters = [5] #number of clusters

# for clust_ind in range(len(number_of_clusters)):
# print("Number of Clusters = ",number_of_clusters[clust_ind])
print("Number of Clusters = ",number_of_clusters[0])
clust_num = number_of_clusters[0]

# kmeans = KMeans(n_clusters=clust_num, random_state=0).fit(X_features_test)
# with config_context(target_offload="gpu:0"):
    # kmeans = KMeans(n_clusters=clust_num, random_state=0, verbose=7).fit(X)

kmeans = KMeans(n_clusters=clust_num, random_state=0, verbose=7).fit(X)
kmean_clust_labels = kmeans.labels_

# np.save('/alina-data1/Zara/Host_clustering/results/5Clusters_Clustering/kmeans/Boruta/new_Labels_kmeans_Boruta.npy', kmean_clust_labels)
np.save(os.getcwd() + '/new_no_feature_selection_Labels_kmeans.npy', kmean_clust_labels)

end_time = time.time() - start_time
print("Clustering Time in seconds =>",end_time)



write_path_112 = os.getcwd() + "/new_no_feature_selection_int_true_variants_k_" + str(clust_num) + ".csv"

with open(write_path_112, 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(0,len(y)):
        ccv = str(y[i])
        writer.writerow([ccv])

write_path_11 = os.getcwd() + "/new_no_feature_selection_orig_true_variants_k_" + str(clust_num) + ".csv"
with open(write_path_11, 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(0,len(y_orig)):
        ccv = str(y_orig[i])
        writer.writerow([ccv])


print("All Processing Done!!!")

