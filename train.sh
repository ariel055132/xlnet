export XLNET_BASE_DIR=../model
export MY_DATASET=../data

python run_classifier.py \
  --use_tpu=False \
  --tpu="" \
  --do_train=True \
  --do_eval=False \
  --eval_all_ckpt=False \
  --task_name=sim \
  --data_dir=$MY_DATASET \
  --output_dir=output/ \
  --model_dir=$XLNET_BASE_DIR \
  --spiece_model_file=$XLNET_BASE_DIR/spiece.model \
  --model_config_path=$XLNET_BASE_DIR/xlnet_config.json \
  --init_checkpoint=$XLNET_BASE_DIR/xlnet_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=2 \
  --eval_batch_size=2 \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --learning_rate=2e-5 \
  --train_steps=4000 \
  --warmup_steps=500 \
  --save_steps=500 \
  --iterations=500
