#!/usr/bin/bash
# CUDA_VISIBLE_DEVICES=2 python train/train_model.py --batch_size 4 --lr 1e-4 #--pretrained_fil /mnt/data/LXY.data/models/head/csr_ShangHai_best.h5
# python utils/h52pb.py

#*** eval acc 
python test/eval.py --trained_model /data/lxy/models/head/csr_ShangHai_best2.h5 --out_file ../logs/testacc.txt \
    --data_dir /data/lxy/shang_crowed