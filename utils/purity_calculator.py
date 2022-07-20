from sklearn.cluster import AgglomerativeClustering
import numpy as np
from utils.logger import CLASS_LABELS_MULTI
import torch

REVERSED_LABELS = dict(map(reversed, CLASS_LABELS_MULTI.items()))

def calculate_purity(c_labels, i, label_idx, label_names):
    label_names = np.array(label_names)
    p_counter = []
    for j in range(i):
        idx = np.where(c_labels==j)
        l_idx = np.intersect1d(idx, label_idx)
        if l_idx.size == 0:
            p_counter.append(0)
        else:
            labels_for_cluster= label_names[l_idx]
            counters = np.unique(labels_for_cluster, return_counts=True)[1]
            maxx =np.max(counters)
            percantage = maxx / np.sum(counters)
            p_counter.append(percantage)
    return np.mean(np.array(p_counter))



def iteration_over_clusters(data, labels, label_idx, start, stop, step):
    r = range(start, stop, step)
    purities  = []
    for i in r:
        cluster = AgglomerativeClustering(n_clusters=i, compute_distances=True)
        cluster = cluster.fit(data)
        
        purities.append(calculate_purity(cluster.labels_, i, label_idx, labels))
    
    return np.array(purities)

def best_cluster_count(purities, start, step):
    return start + step*np.argmax(purities)
        
if __name__ == '__main__':
    x = torch.Tensor([[-0.1154],
        [-0.7554],
        [-1.1980],
        [-1.0964],
        [ 0.9026],
        [-2.1392]])
    n_classes = 3
    labels = ["monkey", "car", "bus", "", "ape", ""]
    label_idx = [0,1,2,4]
    step_size=2
    purities = iteration_over_clusters(n_classes, x, labels, label_idx, step_size)
    print(best_cluster_count(purities, n_classes, step_size))
