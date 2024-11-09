
# 定义保存的路径
model_name="vMLLM_clip_base"
projector_path="save_path/"$model_name"-pretrain"
save_path="save_path/"$model_name"-sft"

# 设置环境
unset http_proxy https_proxy HTTPS_PROXY HTTP_PROXY
export HF_ENDPOINT=https://hf-mirror.com


# # 开始训练
# # pretrain
deepspeed vmllm/train/train_vMLLM_siglip_base.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --version plain \
    --data_path /home/ma-user/work/datasets/blip_laion_cc_sbu_558k/blip_laion_cc_sbu_558k.json \
    --image_folder /home/ma-user/work/datasets/blip_laion_cc_sbu_558k/images \
    --vision_tower google/siglip-base-patch16-224 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $projector_path \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    2>&1 | tee logs/$model_name-pretrain.txt



# sft
deepspeed vmllm/train/train_vMLLM_siglip_base.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --version llama3 \
    --data_path /home/ma-user/work/datasets/yiwei_data/llava_v1_5_sft/llava_v1_5_mix665k.json \
    --image_folder /home/ma-user/work/datasets/yiwei_data/llava_v1_5_sft \
    --vision_tower google/siglip-base-patch16-224 \
    --pretrain_mm_mlp_adapter $projector_path"/mm_projector.bin" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $save_path\
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    2>&1 | tee logs/$model_name-sft.txt
