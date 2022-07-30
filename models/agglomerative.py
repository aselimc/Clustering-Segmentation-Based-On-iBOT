from typing import List
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import mode
import numpy as np
from sklearn.metrics import pairwise_distances
from utils.metrics import mIoU
from . import _BaseSegmentator

class AgglomerativeSegmentator(_BaseSegmentator):
    
    @torch.no_grad()
    def __init__(self,
                backbone,
                logger,
                n_clusters,
                n_chunks,
                linkage='ward',
                distance="euclidean",
                percentage=1.0,
                k=1,
                fit_only_labelled=False,
                **kwargs 
        ):
        super(AgglomerativeSegmentator, self).__init__(backbone, logger, **kwargs)


        self.n_chunks = n_chunks
        self.n_clusters = n_clusters
        self.affinity = distance
        self.linkage = linkage
        self.percentage = percentage
        self.maximum_count_per_class = 1000
        self.global_step =0
        self.k = k
        self.fit_only_labelled = fit_only_labelled
        self.chunked_c_centroids = []
        self.chunked_c_labels= []
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters,
                                             affinity=self.affinity, 
                                             compute_full_tree=False, 
                                             linkage=self.linkage, 
                                             compute_distances=True)
    
    @torch.no_grad()
    def score(self, loader):
        print("Evaluating on different chunks")
        progress_bar = tqdm(total=len(loader))
        top1 = []

        for idx, (images, seg) in enumerate(loader):
            images = images.to(self.device)
            seg = seg.to(self.device)
            preds = []
            for image, gt in zip(images, seg):
                pred = self.forward(image.unsqueeze(0))
                pred = torch.Tensor(mode(pred.cpu(), axis=0)[0]).to(self.device)
                pred = self.smooth(pred.squeeze(0))
                chunk_miou = mIoU(pred, gt)
                preds.append(pred)
                top1.append(chunk_miou)
            pred = torch.stack(preds, dim=0).squeeze(1).squeeze(1)            
            self.logger.log_scalar({'miou':torch.mean(torch.Tensor(top1)).item()}, idx)

            if idx % self.logger.config['eval_freq'] == 0 or idx == len(loader):
                self.logger.log_segmentation(images[0], pred[0], seg[0], step=idx, logit=False)

            self.global_step += 1
            progress_bar.update()
            

        top1 = torch.Tensor(top1)
        miou = torch.mean(top1).item()
        iou_std = torch.std(top1).item()

        self.logger.log_scalar_summary({
                "mIoU": miou,
                "IoU std": iou_std,
            })

        return miou, iou_std


    @torch.no_grad()
    def forward(self, image):
        assert (len(self.chunked_c_centroids) or len(self.chunked_c_labels)),"Data has not been fit!"

        chunk_labels = []
        for centroid, label in zip(self.chunked_c_centroids, self.chunked_c_labels):
            bs = image.size(0)
            centroid = torch.Tensor(centroid).to(self.device)
            label = torch.Tensor(label).to(self.device)
            feat = self._extract_feature(image, flatten=False)
            h = int(np.sqrt(feat.shape[1]))
            feat = feat.flatten(start_dim=0, end_dim=1).permute(0, 1).contiguous()
            pred = self._predict(feat, centroid, label)
            pred = pred.view(bs, self.k, h,h)
            pred = F.interpolate(pred,
                                   size=[self.img_size, self.img_size],
                                   mode='nearest',
                                   recompute_scale_factor=False)
            chunk_labels.append(pred)
        chunk_labels = torch.stack(chunk_labels).squeeze(1)
        chunk_labels = chunk_labels.view(self.n_chunks*self.k, 1, self.img_size, self.img_size)
        return chunk_labels

    def _predict(self, feature, centroids, labels):
        distance = torch.zeros(size=(centroids.shape[0], feature.shape[0])).to(self.device)
        for idx, cluster_centroid in enumerate(centroids):
            distance[idx] = torch.Tensor(pairwise_distances(np.array(feature.cpu()),
                                               np.array(cluster_centroid.cpu()).reshape(1, -1),
                                               metric='cosine')\
                                               .ravel()).to(self.device)
        predict_label = torch.topk(distance.T, self.k, largest=False, dim=1)[1]

        return labels[predict_label]

    @torch.no_grad()
    def fit(self, loader):
        train_features, train_labels = self._transform_data(loader)

        chunked_features = train_features.chunk(self.n_chunks)
        chunked_labels = train_labels.chunk(self.n_chunks)
        progress_bar = tqdm(total=self.n_chunks)
        for feature, label in zip(chunked_features, chunked_labels):
            cluster_centroids, cluster_data_labels = self.fit_chunks(feature, label)
            self.chunked_c_centroids.append(cluster_centroids)
            self.chunked_c_labels.append(cluster_data_labels)
            progress_bar.update()
        self.save_cluster_centroids()

    def fit_chunks(self, feature, label):
        if self.percentage ==0.5:
            ldi = np.arange(len(label)//2)

        elif self.percentage < 1.0:
            label = torch.max(label, dim=1).values
            ldi, label = self._label_equal(label)
            
        else:
            ldi = np.arange(len(label))

        #Not going to be used, only for experimental reasons
        if self.fit_only_labelled:
            model = self.model.fit(feature[ldi])
        else:
            model = self.model.fit(feature)
        cluster_data_labels = self._label_cluster(model, label, ldi)
        cluster_centroids = self._get_cluster_centroids(model, feature)
        del model
        del label
        del ldi
        del feature
        return cluster_centroids, cluster_data_labels

   

    def _get_cluster_centroids(self, model, feature):
        cluster_centroids = []
        for i in range(self.n_clusters):
            cluster_centroids.append(np.mean(np.array(feature[np.where(i==model.labels_)]), axis=0))
        return np.array(cluster_centroids)

    def _label_cluster(self, model, label, ldi):
        cluster_data_labels = np.zeros(shape=(self.n_clusters,), dtype=int)
        for i in range(self.n_clusters):
            # Get the all data index that is inside this cluster
            idx_cluster = np.where(i==model.labels_)
            # Get these the ones where these indexes are labelled
            idx_cluster_labelled = np.intersect1d(idx_cluster, ldi)
            # If there is no labelled data is in this cluster label the   cluster as background
            if len(idx_cluster_labelled) == 0:
                cluster_label = 0
            else:
                classes_inside_cluster, counters = np.unique(label[idx_cluster_labelled], return_counts=True)
                cluster_label = classes_inside_cluster[np.argmax(counters)]
            # # Now assign all datapoints that are inside this cluster as cluster label
            # cluster_data_labels[idx_cluster] = cluster_label
            cluster_data_labels[i] = cluster_label
        return cluster_data_labels
        
    def _label_equal(self, label):
        labelled_data_idx = []
        previous_added_idx = []
        counter_array = torch.zeros(size=(self.num_classes,))
        while len(labelled_data_idx) < int(self.percentage*len(label)):
            random_idx = torch.randint(low=0, high=len(label), size=(1,)).item()
            random_sample = label[random_idx].item()
            if counter_array[random_sample] < self.maximum_count_per_class and random_idx not in previous_added_idx:
                previous_added_idx.append(random_idx)
                counter_array[random_sample] += 1
                labelled_data_idx.append(random_idx)
        ldi=  np.array(labelled_data_idx)
        return ldi, label

    def save_cluster_centroids(self):
        np.save(f'c_centroid_{self.feature}_{self.affinity}_{self.n_chunks}_{self.percentage}_{self.maximum_count_per_class}_{self.n_clusters}.npy', np.array(self.chunked_c_centroids))
        np.save(f'c_label_{self.feature}_{self.affinity}_{self.n_chunks}_{self.percentage}_{self.maximum_count_per_class}_{self.n_clusters}.npy', np.array(self.chunked_c_labels))

    def load_cluster_centroids(self):
        self.chunked_c_centroids = list(np.load(f'c_centroid_{self.feature}_{self.affinity}_{self.n_chunks}_{self.percentage}_{self.maximum_count_per_class}_{self.n_clusters}.npy'))
        self.chunked_c_labels = list(np.load(f'c_label_{self.feature}_{self.affinity}_{self.n_chunks}_{self.percentage}_{self.maximum_count_per_class}_{self.n_clusters}.npy'))
