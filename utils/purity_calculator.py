from sklearn.cluster import AgglomerativeClustering
import numpy as np
from utils.logger import CLASS_LABELS_MULTI
import torch
from scipy.spatial import distance
from tqdm import tqdm

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



def iteration_over_clusters(data, labels, label_idx, start, stop, step, logger):
    r = range(start, stop, step)
    purities  = []
    for i in r:
        cluster = AgglomerativeClustering(n_clusters=i, compute_distances=True)
        cluster = cluster.fit(data)
        p = calculate_purity(cluster.labels_, i, label_idx, labels)
        print(f"# Of Clusters:{i} ---- > Purity: {p}")
        purities.append(p)
    
    return np.array(purities)

def best_cluster_count(purities, start, step):
    return start + step*np.argmax(purities)


def majority_labeller(cluster, n_cluster, label_idx, label_names):
    label_names = np.array(label_names)
    labels = np.empty_like(cluster.labels_, dtype='U100')
    for i in range(n_cluster):
        idx = np.where(cluster.labels_==i)
        l_idx = np.intersect1d(idx, label_idx)
        if len(l_idx) == 0:
            majority = "background"
        else:
            labels_for_cluster= label_names[l_idx]
            classes, counters = np.unique(labels_for_cluster, return_counts=True)
            maxx =np.argmax(counters)
            majority = classes[maxx]
        labels[idx] = majority
    return labels
    
def get_class_means(vit_output, labels):
    mean_dict = dict.fromkeys(REVERSED_LABELS.keys(), np.zeros(shape=(vit_output.shape[1],)))
    for i in REVERSED_LABELS.keys():
        idx = np.where(i == labels)
        idx = np.array(vit_output[idx])
        if len(idx) == 0:
            mean_dict[i] = np.zeros(shape=(vit_output.shape[1],))
        else:
            mean_dict[i] = np.mean(idx, axis=0)
    return mean_dict

def predict(class_means, test):
    image_labels = torch.zeros(size=(test.shape[0], ))
    for idx, patches in enumerate(test):
        label = ""
        d = 9999999
        for key in class_means:
            if key =="contours":
                pass
            dst = distance.euclidean(patches, class_means[key])
            if dst < d:
                d = dst
                label = key
        temp = REVERSED_LABELS[label]
        image_labels[idx] = temp
    h = int(np.sqrt(test.shape[0]))
    image_labels = image_labels.view(1, 1, h,h)
    image_labels = torch.nn.functional.interpolate(image_labels, size=[224, 224])
    return image_labels


def equal_random_selector(real_labels):
    # print("Equal Random Selection Started")
    selected_idx = []
    counter_dict = dict.fromkeys(REVERSED_LABELS, 0)
    maxx = 500

    # print(f"Length of real labels {len(real_labels)}")
    while len(selected_idx) < len(real_labels)//10:
        rs = np.random.randint(low=0, high=len(real_labels))
        random_selection = real_labels[rs]
        if counter_dict[random_selection] < maxx:
            counter_dict[random_selection] += 1
            selected_idx.append(rs)

    return selected_idx
if __name__ == '__main__':
    x = torch.Tensor([[-0.1154],
        [-0.7554],
        [-1.1980],
        [-1.0964],
        [ 0.9026],
        [-2.1392]])
    n_classes = 3
    labels = ["monkey", "car", "bus", "car", "ape", "ape"]
    label_idx = [0,1,2,4]
    step_size=2
    purities = iteration_over_clusters(n_classes, x, labels, label_idx, step_size)
    print(best_cluster_count(purities, n_classes, step_size))
