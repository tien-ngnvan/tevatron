#WANDB_PROJECT='test-uptrain-mBERT' \
#WANDB_API_KEY=0b39e2667775768150d99b18418fb63ca15b13bc \

CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.train \
  --output_dir model_msmarco \
  --model_name_or_path 'checkpoint/checkpoint-10000' \
  --save_steps 20000 \
  --train_dir '/home/gradient/nvtien/datasets/uptrain/uptrain-v2.0/train' \
  --fp16 \
  --streaming \
  --per_device_train_batch_size 8 \
  --train_n_passages 8 \
  --learning_rate 5e-6 \
  --q_max_len 16 \
  --p_max_len 128 \
  --num_train_epochs 1 \
  --logging_steps 500 \
  --overwrite_output_dir \
  --report_to 'wandb'