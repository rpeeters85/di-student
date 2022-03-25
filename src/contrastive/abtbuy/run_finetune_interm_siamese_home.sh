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
TEMP=$5
INTERM=$6
FROZEN=$7
AUG=$8
PREAUG=$9
CUDA_VISIBLE_DEVICES=2 python run_finetune_siamese.py \
	--model_pretrained_checkpoint /work-ceph/rpeeters/phd/dev/di-research/reports/contrastive/abtbuy-clean-$PREAUG$BATCH-$LR-$TEMP-$INTERM-${MODEL##*/}/pytorch_model.bin \
    --do_train \
	--dataset_name=abt-buy \
	--frozen=$FROZEN \
    --train_file /work-ceph/rpeeters/phd/dev/di-research/data/interim/abt-buy/abt-buy-train.json.gz \
	--validation_file /work-ceph/rpeeters/phd/dev/di-research/data/interim/abt-buy/abt-buy-train.json.gz \
	--test_file /work-ceph/rpeeters/phd/dev/di-research/data/interim/abt-buy/abt-buy-gs.json.gz \
	--evaluation_strategy=epoch \
	--tokenizer=$MODEL \
	--grad_checkpoint=$CHECKPOINT \
    --output_dir /work-ceph/rpeeters/phd/dev/di-research/reports/contrastive-ft-siamese/abtbuy-clean-$AUG$BATCH-$PREAUG$LR-$TEMP-$INTERM-$FROZEN-${MODEL##*/}/ \
	--per_device_train_batch_size=64 \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=50 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--metric_for_best_model=loss \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--load_best_model_at_end \
	--augment=$AUG \
	#--do_param_opt \