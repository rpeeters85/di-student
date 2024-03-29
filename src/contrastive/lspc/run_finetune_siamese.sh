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
FROZEN=$6
SIZE=$7
AUG=$8
PREAUG=$9
python run_finetune_siamese.py \
	--model_pretrained_checkpoint /pfs/work7/workspace/scratch/ma_rpeeters-ma_rpeeters-0/phd/dev/di-research/reports/contrastive/computers-$SIZE-$PREAUG$BATCH-$LR-$TEMP-${MODEL##*/}/pytorch_model.bin \
    --do_train \
	--frozen=$FROZEN \
    --train_file /pfs/work7/workspace/scratch/ma_rpeeters-ma_rpeeters-0/phd/dev/di-research/data/interim/wdc-lspc/training-sets/preprocessed_computers_train_$SIZE.pkl.gz \
	--train_size=$SIZE \
	--validation_file /pfs/work7/workspace/scratch/ma_rpeeters-ma_rpeeters-0/phd/dev/di-research/data/interim/wdc-lspc/training-sets/preprocessed_computers_train_$SIZE.pkl.gz \
	--test_file /pfs/work7/workspace/scratch/ma_rpeeters-ma_rpeeters-0/phd/dev/di-research/data/interim/wdc-lspc/gold-standards/preprocessed_computers_gs.pkl.gz \
	--evaluation_strategy=epoch \
	--tokenizer=$MODEL \
	--grad_checkpoint=$CHECKPOINT \
    --output_dir /pfs/work7/workspace/scratch/ma_rpeeters-ma_rpeeters-0/phd/dev/di-research/reports/contrastive-ft-siamese/computers-$SIZE-$AUG$BATCH-$PREAUG$LR-$TEMP-$FROZEN-${MODEL##*/}/ \
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