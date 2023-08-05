curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
git clone https://github.com/hiyouga/LLaMA-Efficient-Tuning.git
conda create -n -y llama_etuning python=3.10
conda activate llama_etuning
#python3 -m venv llm-tuning
#source llm-tuning/bin/activate
cd LLaMA-Efficient-Tuning
pip install -r requirements.txt
mkdir -p ~/sft/llama-7b-lima

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path huggyllama/llama-7b \
    --do_train \
    --dataset lima \
    --template default \
    --finetuning_type lora \
    --output_dir ~/sft/llama-7b-lima \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16


CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path huggyllama/llama-7b \
    --do_train \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type lora \
    --output_dir ~/sft \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16

python src/api_demo.py \
    --model_name_or_path bigscience/bloomz-560m \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint