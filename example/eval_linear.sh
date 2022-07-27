ROOT="/home/rahnfelj"
WEIGHTS="/home/rahnfelj/ibot_weights/ImageNet-22K/ViT-L.pth"
ARCH="vit_large"
PATCH_SIZE="16"
N_BLOCKS="1"
FEATURES=("intermediate" "query" "key" "value")
PERCENTAGE=("0.01" "0.1" "0.5" "1.0")
BACKGROUND_LABEL_PERCENTAGE="1.0"
PATCH_LABELING="coarse"
EPOCHS="50"
WARMUP_EPOCHS="5"
BATCH_SIZE="16"
LR="0.001"
NUM_WORKERS="4"

for feature in ${FEATURES[*]}
do
    for percentage in ${PERCENTAGE[*]}
    do
        python eval_linear.py --root $ROOT --weights $WEIGHTS --arch $ARCH \
        --patch_size $PATCH_SIZE --n_blocks $N_BLOCKS --feature $feature \
        --percentage $percentage --background_label_percentage $BACKGROUND_LABEL_PERCENTAGE \
        --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS --batch_size $BATCH_SIZE --lr $LR \
        --workers $NUM_WORKERS --patch_labeling $PATCH_LABELING --smooth_mask
    done
done
