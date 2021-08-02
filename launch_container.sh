# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
PATH_TO_STORAGE=$1
TXT_DB=$PATH_TO_STORAGE/txt_db
IMG_DIR=$PATH_TO_STORAGE/img_db
OUTPUT=$PATH_TO_STORAGE/finetune
PRETRAIN_DIR=$PATH_TO_STORAGE/pretrained

if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$OUTPUT,dst=/storage,type=bind \
    --mount src=$PRETRAIN_DIR,dst=/pretrain,type=bind,readonly \
    --mount src=$TXT_DB,dst=/txt,type=bind,readonly \
    --mount src=$IMG_DIR,dst=/img,type=bind,readonly \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /src chenrocks/uniter \
