import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchpq.clustering import KMeans
from tqdm import tqdm

from . import _BaseSegmentator
from utils import mIoU
from utils.transforms import PatchwiseSmoothMask


class KMeansSegmentator(_BaseSegmentator):

    def __init__(self, backbone, logger,
                 k=20,
                 distance='cosine',
                 init_mode='kmeans++',
                 n_redo=10,
                 max_iter=300,
                 tol=1e-4,
                 percentage=1.0,
                 weighted_majority_vote=False,
                 fit_clusters=True,
                 arch = "vit_large",
                 extract_vit_features = False,
                 **kwargs):
        super(KMeansSegmentator, self).__init__(backbone, logger, **kwargs)

        self.kmeans = KMeans(n_clusters=k,
                             distance=distance,
                             init_mode=init_mode,
                             n_redo=n_redo,
                             max_iter=max_iter,
                             tol=tol,
                             verbose=5)
        
        self.percentage = percentage
        self.weighted_majority_vote = weighted_majority_vote
        self.fit_clusters = False # change this
        self.arch = arch
        self.extract_vit_features = False # change this
    
        print("percentage", percentage)
        if percentage == 0.01:
            self.maximum_count_per_class = 100
        elif percentage == 0.1:
            self.maximum_count_per_class = 1500
        elif percentage == 0.3:
            self.maximum_count_per_class = 5000
        elif percentage == 0.5:
            self.maximum_count_per_class = 50000

    @torch.no_grad()
    def forward(self, image):
        bs = image.size(0)
        feat = self._extract_feature(image)
        feat = feat.permute(1, 0).contiguous()
        feat = feat.cpu()

        cluster_assignment = self.kmeans.predict(feat)
        cluster_assignment = cluster_assignment.to(device=self.device)
        cluster_assignment = cluster_assignment.view(bs, self.num_patches).unsqueeze(2)
        cluster_assignment = cluster_assignment.expand(bs, self.num_patches, self.patch_size**2).unsqueeze(3)
        cluster_labels = self.cluster_labels.expand(bs, self.num_patches, self.patch_size**2, self.k)
        patch_preds = torch.gather(cluster_labels, dim=3, index=cluster_assignment)
        patch_preds = patch_preds.view(bs, self.num_patches, 1, self.patch_size, self.patch_size)

        # tile label patches together -> (bs, k, 224, 224)
        nrows = self.img_size // self.patch_size
        pred = [make_grid(patch_pred, nrows, padding=0)[0] for patch_pred in patch_preds]
        pred = torch.stack(pred)
        pred = self.smooth(pred)

        return pred

    @torch.no_grad()
    def fit(self, loader):
        if self.extract_vit_features:
            train_features, train_labels = self._transform_data(loader)
            if self.feature == "key":
                torch.save(train_features, "train_features_key.pt")
                torch.save(train_labels, "train_labels_key.pt")
            elif self.feature == "value":
                torch.save(train_features, "train_features_value.pt")
                torch.save(train_labels, "train_labels_value.pt")
            elif self.feature == "query":
                torch.save(train_features, "train_features_query.pt")
                torch.save(train_labels, "train_labels_query.pt")
            elif self.feature == "intermediate":
                torch.save(train_features, "train_features_intermediate.pt")
                torch.save(train_labels, "train_labels_intermediate.pt")
        else:
            print("\nUsing previously extracted features(train_features.pt, train_labels.pt)")
            if self.feature == "key":
                train_features = torch.load("train_features_key.pt")
                train_labels = torch.load("train_labels_key.pt")
            elif self.feature == "value":
                train_features = torch.load("train_features_value.pt")
                train_labels = torch.load("train_labels_value.pt")
            elif self.feature == "query":
                train_features = torch.load("train_features_query.pt")
                train_labels = torch.load("train_labels_query.pt")
            elif self.feature == "intermediate":
                train_features = torch.load("train_features_intermediate.pt")
                train_labels = torch.load("train_labels_intermediate.pt")

        
        train_features_raw = train_features
        train_features = train_features.permute(1, 0).contiguous()
        train_labels = train_labels.long()
        
        # fit clusters, i.e. get centroids (embed_dim, k)
        if self.fit_clusters:
            print("\nFitting clusters...")

            if self.feature == "key":
                self.kmeans.fit(train_features)
                torch.save(self.centroids, 'cluster_centroids_key.pt')
            elif self.feature == "value":
                self.kmeans.fit(train_features)
                torch.save(self.centroids, 'cluster_centroids_value.pt')
            elif self.feature == "query":
                self.kmeans.fit(train_features)
                torch.save(self.centroids, 'cluster_centroids_query.pt')
            elif self.feature == "intermediate":
                self.kmeans.fit(train_features)
                torch.save(self.centroids, 'cluster_centroids_intermediate.pt')
            print("\nCluster centroids saved as cluster_centroids.pt")
        else:
            print("\nUsing previously fitted clusters(cluster_centroids.pt)")
            
            if self.feature == "key":
                loaded_centroids = torch.load('cluster_centroids_key.pt')
            elif self.feature == "value":
                loaded_centroids = torch.load('cluster_centroids_value.pt')
            elif self.feature == "query":
                loaded_centroids = torch.load('cluster_centroids_query.pt')
            elif self.feature == "intermediate":
                loaded_centroids = torch.load('cluster_centroids_intermediate.pt')

            self.kmeans.n_redo = 1
            self.kmeans.max_iter = 300
            self.kmeans.fit(train_features, loaded_centroids)

        # allow only percentage of labels (simulating dataset with small number of labels)
        num_samples = int(train_features.size(1) * self.percentage)
        train_features = train_features[:, :num_samples]
        train_labels = train_labels[:num_samples]

        """if self.percentage == 1.0:
            train_labels = torch.load("train_labels_one_hot.pt")
            print("\nUsing one hot encoded train labels from train_labels_one_hot.pt")
        else:   
            train_labels = F.one_hot(train_labels, self.num_classes)
        """

        # label clusters
        """print("Assigning cluster labels...")
        self.cluster_labels = []
        cluster_assignment = self.kmeans.predict(train_features)
        
        train_features = train_features.to(device=self.device)
        #train_labels = train_labels.to(device=self.device)

        similarities = self._similarity(train_features, self.centroids.to(device=self.device))

        if self.feature == "key":
            torch.save(cluster_assignment, "cluster_assignment_key.pt")
            print("\nCluster assignment saved")
            torch.save(similarities, "similarities_key.pt")
            print("\nSimilarities saved")
        elif self.feature == "value":
            torch.save(cluster_assignment, "cluster_assignment_value.pt")
            print("\nCluster assignment saved")
            torch.save(similarities, "similarities_value.pt")
            print("\nSimilarities saved")
        elif self.feature == "query":
            torch.save(cluster_assignment, "cluster_assignment_query.pt")
            print("\nCluster assignment saved")
            torch.save(similarities, "similarities_query.pt")
            print("\nSimilarities saved")
        elif self.feature == "intermediate":
            torch.save(cluster_assignment, "cluster_assignment_intermediate.pt")
            print("\nCluster assignment saved")
            torch.save(similarities, "similarities_intermediate.pt")
            print("\nSimilarities saved. FINISHED...")

        for idx in range(self.k):
            assigned_train_labels = train_labels[cluster_assignment == idx]

            # assign background to clusters with no labeled data
            if assigned_train_labels.size(0) == 0:
                label = torch.zeros(self.patch_size**2).to(device=self.device)
            else:
                if self.weighted_majority_vote:
                    # higher similarity -> higher weight
                    assigned_similarities = similarities[cluster_assignment == idx, idx]
                    weights = torch.softmax(assigned_similarities, dim=0)
                    weights = weights.unsqueeze(1).unsqueeze(2)
                else:
                    weights = 1.0

                assigned_train_labels = train_labels[cluster_assignment == idx]
                vote = torch.sum(weights * assigned_train_labels, dim=0)
                label = torch.argmax(vote, dim=1)
            
            self.cluster_labels.append(label)

        self.cluster_labels = torch.stack(self.cluster_labels, dim=1).unsqueeze(0).unsqueeze(0)"""
        if self.feature == "key":
           self.cluster_labels = torch.load("cluster_labels_key.pt").cuda()
           print("\nUsing pre-computed cluster labels cluster_labels_key.pt")
        elif self.feature == "value":
           self.cluster_labels = torch.load("cluster_labels_value.pt").cuda()
           print("\nUsing pre-computed cluster labels cluster_labels_value.pt")
        elif self.feature == "query":
           self.cluster_labels = torch.load("cluster_labels_query.pt").cuda()
           print("\nUsing pre-computed cluster labels cluster_labels_query.pt")
        elif self.feature == "intermediate":
           self.cluster_labels = torch.load("cluster_labels_intermediate.pt").cuda()
           print("\nUsing pre-computed cluster labels cluster_labels_intermediate.pt")

    def _label_equal(self, label):
        labelled_data_idx = []
        previous_added_idx = []
        counter_array = torch.zeros(size=(self.num_classes,))
        print("\nBalancing data...")
        progress_bar = tqdm(total=int(self.percentage*len(label)))
        while len(labelled_data_idx) < int(self.percentage*len(label)):
            random_idx = torch.randint(low=0, high=len(label), size=(1,)).item()
            random_sample = label[random_idx].item()
            if counter_array[random_sample] < self.maximum_count_per_class and random_idx not in previous_added_idx:
                previous_added_idx.append(random_idx)
                counter_array[random_sample] += 1
                labelled_data_idx.append(random_idx)
                progress_bar.update()
        ldi=  np.array(labelled_data_idx)
        return ldi, label

    def _similarity(self, x, y, inplace=False, normalize=True):
        return self.kmeans.sim(x, y, inplace, normalize)

    @property
    def centroids(self):
        return self.kmeans.centroids
    
    @property
    def k(self):
        return self.kmeans.n_clusters

    @property
    def distance(self):
        return self.kmeans.distance