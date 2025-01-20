# export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export PRETRAINED_BLIP_DIFFUSION="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP-Diffusion/blip-diffusion.tar.gz"
# export INSTANCE_DIR="/shenzihe/dreambooth/dataset/cat"
# export OUTPUT_DIR="/shenzihe/WD-Disk/output/exp"

# accelerate launch train.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --pretrained_BLIPdiffusion_name_or_path=$PRETRAINED_BLIP_DIFFUSION \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --subject_text="cat" \
#   --text_prompt="a cat with <p> pose in <b> background" \
#   --questions='{"<p>": "what is the pose for the cat?", "<b>": "what is the background for the cat"}' \
#   --resolution=512 \
#   --train_batch_size=3 \
#   --gradient_accumulation_steps=1 \
#   --checkpointing_steps=100 \
#   --learning_rate=1e-4 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=100 \
#   --validation_prompt="A cat in the jungle" \
#   --validation_epochs=100 \
#   --seed=42 \
#   --loss=0.0 \
#   --loss_1=1.0 \
#   --loss_2=0.0 \

#!/bin/bash
#!/bin/bash

# 设置模型和预训练路径
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export PRETRAINED_BLIP_DIFFUSION="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP-Diffusion/blip-diffusion.tar.gz"


# 设置实例和输出目录
export INSTANCE_DIR="/shenzihe/multi_BlipDisenBooth/dataset/category/dog6_and_dog2"
export OUTPUT_DIR="/shenzihe/WD-Disk/output/EXP1/dog6_and_dog2"

# 设置主题文本和提示，使用 subject_name
SUBJECT_A_TEXT="corgi"
SUBJECT_B_TEXT="chow chow"
TEXT_PROMPT="a $SUBJECT_A_TEXT with <p_A> pose and a $SUBJECT_B_TEXT with <p_B> pose in <b> background"

# 构建 JSON 格式的 questions，使用单引号包裹，内部使用双引号
QUESTIONS='{"<p>": "what is the pose?", "<b>": "what is the background?"}'

# 设置验证提示
VALIDATION_PROMPT="A $subject_name in the jungle"

# 输出当前正在训练的主题
echo "正在训练主题：$subject_name（类别：$class）"

# 运行训练命令
accelerate launch train.py \
  --pretrained_model_name_or_path="$MODEL_NAME"  \
  --pretrained_BLIPdiffusion_name_or_path="$PRETRAINED_BLIP_DIFFUSION" \
  --instance_data_dir="$INSTANCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --subject_text "$SUBJECT_A_TEXT" "$SUBJECT_B_TEXT" \
  --text_prompt="$TEXT_PROMPT" \
  --questions="$QUESTIONS" \
  --resolution=512 \
  --train_batch_size=3 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100 \
  --validation_prompt="$VALIDATION_PROMPT" \
  --validation_epochs=100 \
  --seed=42 \
  --ctx_begin_pos=78

echo "完成主题 $subject_name 的训练"
echo "---------------------------------------------"

# # 定义主题列表，每行包含 subject_name 和 class，用逗号分隔
# subjects=(
# "backpack,backpack"
# "backpack_dog,backpack"
# "bear_plushie,stuffed animal"
# "berry_bowl,bowl"
# "can,can"
# "candle,candle"
# "cat,cat"
# "cat2,cat"
# "clock,clock" 
# "colorful_sneaker,sneaker"
# "dog,dog"
# "dog2,dog"
# "dog3,dog"
# "dog5,dog"
# "dog6,dog"
# "dog7,dog"
# "dog8,dog"
# "duck_toy,toy"
# "fancy_boot,boot"
# "grey_sloth_plushie,stuffed animal"
# "monster_toy,toy"
# "pink_sunglasses,glasses"
# "poop_emoji,toy"
# "rc_car,toy"
# "red_cartoon,cartoon"
# "robot_toy,toy"
# "shiny_sneaker,sneaker"
# "teapot,teapot"
# "vase,vase"
# "wolf_plushie,stuffed animal"
# )

# # 遍历每个主题进行训练
# for item in "${subjects[@]}"; do
#   # 使用 IFS 和 read 将 subject_name 和 class 分开
#   IFS=',' read -r subject_name class <<< "$item"

#   # 设置实例和输出目录
#   export INSTANCE_DIR="/shenzihe/dreambooth/dataset/$subject_name"
#   export OUTPUT_DIR="/shenzihe/WD-Disk/output/exp6/$subject_name"

#   # 设置主题文本和提示，使用 subject_name
#   SUBJECT_A_TEXT="$subject_name"
#   SUBJECT_B_TEXT=""
#   TEXT_PROMPT="a $subject_name with <pA> pose and a  in <b> background"

#   # 构建 JSON 格式的 questions，使用单引号包裹，内部使用双引号
#   QUESTIONS='{"<p>": "what is the pose?", "<b>": "what is the background?"}'

#   # 设置验证提示
#   VALIDATION_PROMPT="A $subject_name in the jungle"

#   # 输出当前正在训练的主题
#   echo "正在训练主题：$subject_name（类别：$class）"

#   # 运行训练命令
#   accelerate launch train.py \
#     --pretrained_model_name_or_path="$MODEL_NAME"  \
#     --pretrained_BLIPdiffusion_name_or_path="$PRETRAINED_BLIP_DIFFUSION" \
#     --instance_data_dir="$INSTANCE_DIR" \
#     --output_dir="$OUTPUT_DIR" \
#     --subject_text="$SUBJECT_A_TEXT" "$SUBJECT_B_TEXT" \
#     --text_prompt="$TEXT_PROMPT" \
#     --questions="$QUESTIONS" \
#     --resolution=512 \
#     --train_batch_size=6 \
#     --gradient_accumulation_steps=1 \
#     --checkpointing_steps=100 \
#     --learning_rate=1e-4 \
#     --lr_scheduler="constant" \
#     --lr_warmup_steps=0 \
#     --max_train_steps=100 \
#     --validation_prompt="$VALIDATION_PROMPT" \
#     --validation_epochs=100 \
#     --seed=42 \
#     --ctx_begin_pos=78

#   echo "完成主题 $subject_name 的训练"
#   echo "---------------------------------------------"

# done


