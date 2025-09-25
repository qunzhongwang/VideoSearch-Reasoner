
export WANDB_PROJECT=vlm-r1-sft

export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export WANDB_API_KEY=""
export HF_HOME=""

RUN_NAME=""
export LOG_PATH="./debug_log_$RUN_NAME.txt"


python -m torch.distributed.run --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12347" \
    cs_sft_train/sft_tool.py \
    --deepspeed config/zero3.json \
    --output_dir cs_sft_train/output/$RUN_NAME \
    --model_name_or_path  "" \
    --datasetpath "" \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --eval_strategy no \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --bf16 \
    --learning_rate 1e-6 \
    --torch_dtype bfloat16 \
    --data_seed 49 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 12 \
    --run_name $RUN_NAME \
    --save_strategy steps \
    --save_steps 20 \
    --save_only_model true \
    --freeze_vision_modules true  \
    --save_total_limit 1


    