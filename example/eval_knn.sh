ROOT="/home/rahnfelj"
WEIGHTS="/home/rahnfelj/ibot_weights/ImageNet-22K/ViT-L.pth"
ARCH="vit_large"
PATCH_SIZE="16"
N_BLOCKS="1"
FEATURES=("intermediate" "query" "key" "value")
PERCENTAGE=("0.01" "0.1" "0.5" "1.0")
BACKGROUND_LABEL_PERCENTAGE="1.0"
N_NEIGHBORS="20"
BATCH_SIZE="16"
NUM_WORKERS="4"
PATCH_LABELING="coarse"

for feature in ${FEATURES[*]}
do
    for percentage in ${PERCENTAGE[*]}
    do
        python eval_knn.py --root $ROOT --weights $WEIGHTS --arch $ARCH \
        --patch_size $PATCH_SIZE --n_blocks $N_BLOCKS --feature $feature \
        --percentage $percentage --background_label_percentage $BACKGROUND_LABEL_PERCENTAGE \
        --n_neighbors $N_NEIGHBORS --batch_size $BATCH_SIZE --workers $NUM_WORKERS \
        --patch_labeling $PATCH_LABELING
    done
done
