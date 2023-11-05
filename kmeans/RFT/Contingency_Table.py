import numpy as np
import pandas as pd
import csv
import os

#Variants = pd.read_csv("new_attributes_freq_vec_hard_kmeans_clustering_k_22.csv", header=None)
#Cluster_ids = pd.read_csv("new_freq_vec_hard_kmeans_clustering_k_22.csv", header=None)


read_path = os.getcwd() + "/new_orig_true_variants_k_5.csv"
Variants = []
with open(read_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        tmp = str(row)
        tmp_1 = tmp.replace("[","")
        tmp_2 = tmp_1.replace("]","")
        tmp_3 = tmp_2.replace("\'","")
        Variants.append(tmp_3)



# read_path = "/alina-data1/Zara/Host_clustering/results/5Clusters_Clustering/kmeans/RFT/new_Labels_kmeans_RFT.csv"
read_path = os.getcwd() + "/new_Labels_kmeans_RFT.npy"
Cluster_ids = np.load(read_path)

print("data loaded")
unique_varaints = list(np.unique(Variants))
print(unique_varaints)
int_var = []
for i in range(0,len(Variants)):
    temp_var = Variants[i]
#    print("variant = ",temp_var)
    temp_index = unique_varaints.index(temp_var)
#    print("temp_index = ",temp_index)
    int_var.append(temp_index)
print("preprocessing done")

s = (len(unique_varaints),5)
# s = (len(unique_varaints),len(np.unique(Cluster_ids)))
cnt = np.array(np.zeros(s))
for i in range(len(Variants)):
#    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ",int_var[i]," ----------   ",Cluster_ids[i])
    int_1 = int(int_var[i])
    int_2 = int(Cluster_ids[i])
    cnt[int_1,int_2] = cnt[int_1,int_2] + 1

write_path_112 = os.getcwd() + "/cnt_kmean_RFT_5cluster.csv"

with open(write_path_112, 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(0,len(cnt)):
        ccv = list(cnt[i])
        writer.writerow(ccv)

print("Done")








#data_crosstab = pd.crosstab(index=unique_varaints, columns=freq)
#print(data_crosstab)
#with open('unique_class_lable.npy', 'wb') as f:
#    np.save(f, np.array(unique_varaints))
#np.save('unique_class_lable.npy', unique_varaints)
