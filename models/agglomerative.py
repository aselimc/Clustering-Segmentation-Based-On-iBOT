from typing import List
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import cluster
from scipy.stats import mode
import numpy as np
from sklearn.metrics import pairwise_distances
from utils.metrics import mIoU

class AgglomerativeClustering(nn.Module):
    
    def __init__(self,
                backbone,
                logger,
                n_clusters,
                n_chunks,
                feature='intermediate',
                n_blocks=1,
                use_cuda=True,
                distance="euclidean",
                calculate_purity=False,
                temperature=1.0,
                patch_labeling='coarse',
                n_classes=21,
                affinity='cosine',
                linkage='average',
                percentage=1.0    
        ):
        super(AgglomerativeClustering, self).__init__()
        self.device = torch.device('cuda') if use_cuda else torch.device('cpu')
        self.backbone = backbone.to(device=self.device)
        self.feature = feature
        self.n_blocks = n_blocks
        self.n_chunks = n_chunks
        self.num_classes = n_classes
        self.logger = logger
        self.patch_labeling = patch_labeling
        self.temperature = temperature
        self.calculate_purity = calculate_purity
        self.distance = distance
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.linkage = linkage
        self.percentage = percentage
        self.maximum_count_per_class = 200
        self.chunked_c_centroids : List
        self.chunked_c_labels: List
        self.global_step =0

    @torch.no_grad()
    def forward(self, loader):
        print("Evaluating on different chunks")
        progress_bar = tqdm(total=len(loader))
        image_miou = []
        for idx, (image, seg) in enumerate(loader):
            image = image.cuda()
            predictions = np.stack(self.predict(image), axis=0).squeeze(1).astype('int64')
            pred = mode(predictions, axis=0)[0].astype('int64')
            pred = torch.Tensor(pred)
            chunk_miou = mIoU(pred, seg, no_softmax=False)
            self.logger.log_scalar({'chunk_miou':chunk_miou}, self.global_step)
            if idx % 5 ==0:
                    self.logger.log_cluster_segmentation(image[0], pred, seg, self.global_step)
            self.global_step += 1
            progress_bar.update()
            image_miou.append(chunk_miou)
        image_miou = np.array(image_miou)


    @torch.no_grad()
    def predict(self, image):
        if not(len(self.chunked_c_centroids) or len(self.chunked_c_labels)):
            print("Data has not been fit!")
            pass
        chunk_labels = []
        for centroid, label in zip(self.chunked_c_centroids, self.chunked_c_labels):
            bs = image.size(0)
            feat = self._extract_feature(image)
            h = int(np.sqrt(feat.shape[1]))
            feat = feat.flatten(start_dim=0, end_dim=1).permute(0, 1).contiguous()
            feat = feat.cpu()
            image_labels = self._predict(feat, centroid, label)
            image_labels = image_labels.view(bs, 1, h,h)
            image_labels = torch.nn.functional.interpolate(image_labels, size=[224, 224])
            chunk_labels.append(image_labels)
        return chunk_labels

    @torch.no_grad()
    def fit(self, loader):
        train_features = []
        train_labels = []
        self.cluster_labels = []

        print("Extracting ViT features...")
        progress_bar = tqdm(total=len(loader))
        for image, target in loader:
            image = image.to(device=self.device)
            feat = self._extract_feature(image)
            feat = feat.flatten(start_dim=0, end_dim=1).cpu()
            train_features.append(feat)

            # divide ground truth mask into patches
            target = target.to(device=self.device)
            target = self.unfold(target.unsqueeze(1).float())
            target = target.permute(0, 2, 1)
            if self.patch_labeling == 'coarse':
                target = target.long()
                target = F.one_hot(target, self.num_classes)
                target = torch.argmax(target.sum(dim=2), dim=2)
                target = target.unsqueeze(2).expand(-1, self.num_patches, self.patch_size**2)
            target = target.flatten(start_dim=0, end_dim=1)
            target = target.byte().cpu()
            train_labels.append(target)
            progress_bar.update()
        progress_bar.close()
        print("\nFitting chunks into clusters...")
        train_features = torch.cat(train_features, dim=0).permute(0, 1).contiguous()
        train_labels = torch.cat(train_labels, dim=0).long()
        chunked_features = train_features.chunk(self.n_chunks)
        chunked_labels = train_labels.chunk(self.n_chunks)
        progress_bar = tqdm(total=self.n_chunks)
        chunked_c_centroids = []
        chunked_c_labels = []
        for feature, label in zip(chunked_features, chunked_labels):
            cluster_centroids, cluster_data_labels = self.fit_chunks(feature, label)
            chunked_c_centroids.append(cluster_centroids)
            chunked_c_labels.append(cluster_data_labels)
            progress_bar.update()
        self.chunked_c_centroids = chunked_c_centroids
        self.chunked_c_labels = chunked_c_labels
        self.save_cluster_centroids()

    def fit_chunks(self, feature, label):
        if self.percentage < 1.0:
            label = torch.max(label, dim=1).values
            ldi, label = self._label_equal(label)
        else:
            ldi = np.arange(len(label))
        model = cluster.AgglomerativeClustering(n_clusters=self.n_clusters, affinity=self.affinity,
                                      compute_full_tree=False, linkage=self.linkage, compute_distances=True)
        model = model.fit(feature)
        cluster_data_labels = self._label_cluster(model, label, ldi)
        cluster_centroids = self._get_cluster_centroids(model, feature)
        del model
        del label
        del ldi
        del feature
        return cluster_centroids, cluster_data_labels

    def _predict(self, feature, centroids, labels):
        distance = np.zeros(shape=(centroids.shape[0], feature.shape[0]))
        for idx, cluster_centroid in enumerate(centroids):
            distance[idx] = pairwise_distances(feature, cluster_centroid.reshape(1, -1), metric="cosine").ravel()
        predict_label = np.argmin(distance.T, axis=1)

        return torch.Tensor(labels[predict_label])     

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

    def _extract_feature(self, images):
        if self.feature == 'intermediate':
            intermediate_output = self.backbone.get_intermediate_layers(images, self.n_blocks)
        else:
            intermediate_output = self.backbone.get_qkv(images, self.n_blocks, out=self.feature)
        feat = torch.stack(intermediate_output, dim=2)
        feat = torch.mean(feat, dim=2)
        feat = feat[:, 1:]

        return feat

    def save_cluster_centroids(self):
        np.save('c_centroid_cosine.npy', np.array(self.chunked_c_centroids))
        np.save('c_label_cosine.npy', np.array(self.chunked_c_labels))

    def load_cluster_centroids(self):
        self.chunked_c_centroids = list(np.load('c_centroid_cosine.npy'))
        self.chunked_c_labels = list(np.load('c_label_cosine.npy'))

    @property
    def patch_size(self):
        return self.backbone.patch_embed.patch_size

    @property
    def img_size(self):
        return self.backbone.patch_embed.img_size

    @property
    def num_patches(self):
        return int((self.img_size / self.patch_size)**2)