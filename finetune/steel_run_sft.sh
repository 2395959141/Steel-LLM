##########################
# llama factory start script
##########################
#CUDA_VISIBLE_DEVICES=5,6 python ../LLaMA-Factory/src/train.py \
export LLaMA_PATH=/home/chenyuhang/LLaMA-Factory
OUTPUT_DIR=/DATA/disk2/yuhang/.cache/ckpt/steel_llm/sft_ckpt
export CUDA_VISIBLE_DEVICES=4,5,6,7 
#python  $LLaMA_PATH/src/train.py \
# max_steps / num_train_epochs
# --streaming True \
# 6卡配置：1,2,3,4,5,7
# 原始ckpt：--model_name_or_path /data/model/llm/Steel-LLM/steel-llm-step-1060000-ckpt \
# all data: 
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 $LLaMA_PATH/src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path /home/chenyuhang/Steel-LLM/pretrain_modify_from_TinyLlama/model/steel_modify_from_qwen_1_5 \
    --cutoff_len 768 \
    --dataset_dir /DATA/disk2/yuhang/.cache/steel_dataset/sft_data/llamafactory_input \
    --dataset baai_instruct_7m,code_feedback_custom,openhermes_custom \
    --report_to wandb \
    --run_name sft_120000 \
    --preprocessing_num_workers 72 \
    --template qwen \
    --finetuning_type full \
    --output_dir ${OUTPUT_DIR}/Code-Feedback_Infinity-Instruct_7M_OpenHermes \
    --overwrite_output_dir \
    --overwrite_cache \
    --per_device_train_batch_size 26 \
    --per_device_eval_batch_size 4 \
    --do_eval \
    --val_size 0.01 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --flash_attn fa2\
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 2950 \
    --learning_rate 3e-5 \
    --weight_decay 0.0 \
    --num_train_epochs 5.0 \
    --plot_loss \
    --bf16


    #--evaluation_strategy 5000 \