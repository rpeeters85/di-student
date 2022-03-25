#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
MODEL=$1
CHECKPOINT=$2
BATCH=$3
LR=$4
TEMP=$5
AUG=$6
CUDA_VISIBLE_DEVICES=2 python run_pretraining_deepmatcher.py \
    --do_train \
	--dataset_name=beeradvo-ratebeer \
	--clean=True \
    --train_file /work-ceph/rpeeters/phd/dev/di-research/data/processed/beeradvo-ratebeer/contrastive/beeradvo-ratebeer-train.pkl.gz \
	--id_deduction_set /work-ceph/rpeeters/phd/dev/di-research/data/interim/beeradvo-ratebeer/beeradvo-ratebeer-train.json.gz \
	--tokenizer=$MODEL \
	--grad_checkpoint=$CHECKPOINT \
    --output_dir /work-ceph/rpeeters/phd/dev/di-research/reports/contrastive/dev-beeradvoratebeer-clean-$AUG$BATCH-$LR-$TEMP-${MODEL##*/}/ \
	--temperature=$TEMP \
	--per_device_train_batch_size=$BATCH \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--logging_strategy="epoch" \
	--augment=$AUG \
	#--dataloader_num_workers=4 \