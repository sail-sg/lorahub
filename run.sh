export LORA_RANK=16
export OUTPUT_PREFIX=checkpoints_flan_t5_large

for file in "flan_task"/*; do
  if [ -f "$file" ]; then
    filename=$(basename "$file")
    
    # check if : in the file name
    if [[ "$filename" == *:* ]]; then
      prefix="${filename%%:*}"
    else
      prefix="${filename%%.*}"
    fi
    echo "$OUTPUT_PREFIX/$prefix"
    if [ ! -d "$OUTPUT_PREFIX/$prefix" ]; then
    echo "train lora modules on $filename"
    accelerate launch train_model.py \
          --model_name_or_path google/flan-t5-large \
          --dataset_name $file \
          --input_column inputs \
          --output_column targets \
          --do_train \
          --do_eval \
          --per_device_train_batch_size 10 \
          --per_device_eval_batch_size 48 \
          --gradient_accumulation_steps 1 \
          --learning_rate 2e-4 \
          --preprocessing_num_workers 16 \
          --generation_max_length 256 \
          --logging_strategy steps \
          --logging_steps 10 \
          --num_train_epochs 20 \
          --lora_r $LORA_RANK \
          --evaluation_strategy epoch \
          --save_strategy epoch \
          --metric_for_best_model exact_match \
          --predict_with_generate \
          --warmup_steps 0 \
          --max_seq_length 1024 \
          --max_answer_length 256 \
          --val_max_answer_length 256 \
          --save_total_limit 5 \
          --output_dir $OUTPUT_PREFIX/$prefix \
          --run_name "$OUTPUT_PREFIX_$prefix"
     fi
  fi
done