#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --export=NONE
MODEL=$1
CHECKPOINT=$2
BATCH=$3
LR=$4
AUG=$5
CUDA_VISIBLE_DEVICES=4 python run_finetune_baseline.py \
    --do_train \
	--dataset_name=abt-buy \
    --train_file /work-ceph/rpeeters/phd/dev/di-research/data/interim/abt-buy/abt-buy-train.json.gz \
	--validation_file /work-ceph/rpeeters/phd/dev/di-research/data/interim/abt-buy/abt-buy-train.json.gz \
	--test_file /work-ceph/rpeeters/phd/dev/di-research/data/interim/abt-buy/abt-buy-gs.json.gz \
	--evaluation_strategy=epoch \
	--tokenizer=$MODEL \
	--grad_checkpoint=$CHECKPOINT \
    --output_dir /work-ceph/rpeeters/phd/dev/di-research/reports/baseline/dev-abtbuy-$AUG$BATCH-$LR-${MODEL##*/}/ \
	--per_device_train_batch_size=$BATCH \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--metric_for_best_model=loss \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--load_best_model_at_end \
	--augment=$AUG \
	#--fp16 \
	#--do_param_opt \
	#--dataloader_num_workers=4 \