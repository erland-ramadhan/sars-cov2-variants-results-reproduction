import pandas as pd
import numpy as np
import time
from sklearnex import patch_sklearn
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

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from sklearn.decomposition import TruncatedSVD
# import random
# import seaborn as sns
import os.path as path
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
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn import metrics
# from sklearn import svm

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn import metrics

import pandas as pd
import sklearn
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import classification_report, confusion_matrix

# from sklearn.neighbors import KNeighborsClassifier

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix

# from pandas import DataFrame

# from sklearn.model_selection import KFold
# from sklearn.model_selection import RepeatedKFold

# from sklearn.metrics import confusion_matrix

# from numpy import mean
# from sklearn.linear_model import LogisticRegression
#from boruta import BorutaPy
#from sklearn.ensemble import RandomForestClassifier
#import seaborn as sns




## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# matplotlib.rcParams['mathtext.fontset'] = 'cm'

## for LaTeX typefont
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'

## for another LaTeX typefont
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

# rc('text', usetex = True)

print("Packages Loaded!!!")


# # Read Frequency Vector

# In[ ]:


vector_path = os.getcwd() + '/../../feature_frequency_vector.npy'
variant_path = os.getcwd() + '/../../filtered_variants.npy'




X = np.load(vector_path)
y_orig = np.load(variant_path)




# frequency_vector_read_final = np.load("/alina-data1/sarwan/IEEE_BigData/Dataset/Complete_all_freq_vec_data_kmers.npy")

# print("Frequency Vector Data Reading Done with length ==>>",len(frequency_vector_read_final))

# #print("Frequency Vector integer conversion Done")



# read_path = "/alina-data1/sarwan/IEEE_BigData/Dataset/Complete Clustering Data/complete_other_attributes_only.csv"

# variant_orig = []

# with open(read_path) as csv_file:
    # csv_reader = csv.reader(csv_file, delimiter=',')
    # for row in csv_reader:
        # tmp = row
        # variant_orig.append(tmp[1])

print("Attributed data Reading Done")


# In[14]:


unique_varaints = list(np.unique(y_orig))


# In[18]:


int_variants = []
for ind_unique in range(len(y_orig)):
    variant_tmp = y_orig[ind_unique]
    ind_tmp = unique_varaints.index(variant_tmp)
    int_variants.append(ind_tmp)

print("Attribute data preprocessing Done")


######################################################################
# tem_fre_vec = frequency_vector_read_final[0:500]
# tmp_true_labels = int_variants[0:500]
# tmp_true_labels_orig = variant_orig[0:500]

# X = np.array(tem_fre_vec)
# y =  np.array(tmp_true_labels)
# y_orig =  np.array(tmp_true_labels_orig)
######################################################################

# X = np.array(frequency_vector_read_final)
y =  np.array(int_variants, dtype=np.uint8)
# y_orig = np.array(variant_orig)

#ridge_data = pd.DataFrame(X)

from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit
sss = ShuffleSplit(n_splits=1, test_size=0.9)
sss.get_n_splits(X, y)
train_index, test_index = next(sss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
y_train_orig, y_test_orig = y_orig[train_index], y_orig[test_index]

print("Train-Test Split Done")


print("X_train rows = ",X_train.shape[0],"X_train columns = ",X_train.shape[1])
print("X_test rows = ",X_test.shape[0],"X_test columns = ",X_test.shape[1])

print("Lasso Features Starts here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# start_time = time.time()

# rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
# # define Boruta feature selection method
# feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=1)
# # find all relevant features - 5 features should be selected
# feat_selector.fit(X_train, y_train)
# # call transform() on X to filter it down to selected features
# X_features_test = feat_selector.transform(frequency_vector_read_final)


####################################################################################
#sel_ = SelectFromModel(LogisticRegression(penalty='l1', C=1, solver='liblinear'))
#sel_.fit(X_train, y_train)
#X_features_train = sel_.transform(X_train)
#X_features_test = sel_.transform(X)


scaler = StandardScaler()

# L1 = Lasso, L2 = Ridge
sel_ = SelectFromModel(estimator=LogisticRegression(n_jobs=-1, penalty='l1', solver='liblinear'))
sel_.fit(scaler.fit_transform(X_train), y_train)

print('total features: {}'.format(X_train.shape[1]))
print('selected features: {}'.format(np.sum(sel_.get_support() == True)))
print('features with coefficients shrank to zero: {}'.format(
      np.sum(sel_.estimator_.coef_ == 0)))

X_features_test = sel_.transform(X)
######################################################################################

start_time = time.time()

#for clustering, the input data is in variable X_features_test
from sklearn.cluster import KMeans


number_of_clusters = [5] #number of clusters

for clust_ind in range(len(number_of_clusters)):
    print("Number of Clusters = ",number_of_clusters[clust_ind])
    clust_num = number_of_clusters[clust_ind]

    # with config_context(target_offload="gpu:0"):
    #     kmeans = KMeans(n_clusters=clust_num, random_state=0).fit(X_features_test)

    kmeans = KMeans(n_clusters=clust_num, random_state=0).fit(X_features_test)
    kmean_clust_labels = kmeans.labels_

    np.save(os.getcwd() + '/new_Labels_kmeans_Lasso.npy', kmean_clust_labels)


    end_time = time.time() - start_time
    print("Clustering Time in seconds =>",end_time)



    write_path_112 = os.getcwd() + "/new_int_true_variants_k_" + str(clust_num) + ".csv"

    with open(write_path_112, 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(0,len(y)):
            ccv = str(y[i])
            writer.writerow([ccv])

    write_path_11 = os.getcwd() + "/new_orig_true_variants_k_" + str(clust_num) + ".csv"
    with open(write_path_11, 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(0,len(y_orig)):
            ccv = str(y_orig[i])
            writer.writerow([ccv])


print("All Processing Done!!!")

