import numpy as np
import pandas as pd
import os
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.metrics import f1_score

calculated_cluster = np.load('new_Labels_fuzzy_RFT.npy')

orig_df = pd.read_csv('new_int_true_variants_k_5.csv', header=None)
orig_data = orig_df.to_numpy(dtype=np.uint8)

variants = np.load(os.getcwd() + "/../../filtered_variants.npy")
variants_label = np.unique(variants)

cluster_var_label = np.empty(variants_label.shape[0], dtype='U9')
cluster_var = np.empty(variants.shape, dtype='U9')

yet = list(range(variants_label.shape[0]))
done_var = []
done_id = []

while "" in cluster_var_label:
    i = yet.pop(0)
    # print("yet is ", yet)
    id = orig_data[np.where(calculated_cluster == i)[0]]
    value_id_, count_id_ = np.unique(id, return_counts=True)

    value_id = np.array(range(variants_label.shape[0]))
    count_id = np.zeros(variants_label.shape[0])
    count_id[value_id_] = count_id_

    value_id = np.delete(value_id, done_var)
    count_id = np.delete(count_id, done_var)

    id_max = value_id[np.argmax(count_id)]

    var = calculated_cluster[np.where(orig_data == id_max)[0]]
    value_var_, count_var_ = np.unique(var, return_counts=True)

    value_var = np.array(range(variants_label.shape[0]))
    count_var = np.zeros(variants_label.shape[0])
    count_var[value_var_] = count_var_
    # print("done is ", done_var)

    if np.max(count_var) ==  np.max(count_id):
        cluster_var_label[i] = variants_label[id_max]
        done_var.append(id_max)
        done_id.append(i)
        # done.append(i)

    else:
        value_var = np.delete(value_var, done_id)
        count_var = np.delete(count_var, done_id)
        if np.max(count_var) == np.max(count_id):
            cluster_var_label[i] = variants_label[id_max]
            done_var.append(id_max)
            done_id.append(i)
            # done.append(i)
        else:
            yet.append(i)

    # print(cluster_var_label)

for i in cluster_var_label:
    cluster_id = np.where(cluster_var_label == i)[0][0]
    cluster_var[np.where(calculated_cluster == cluster_id)] = [i]

for i in cluster_var_label:
    actual = variants[variants == i]
    predicted = cluster_var[np.where(variants == i)[0]][:,0]
    print("Weighted F1 Score for {} is {}".format(i, f1_score(actual, predicted, average='weighted')))
