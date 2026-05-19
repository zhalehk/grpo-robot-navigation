# GRPO Robot Navigation Training

Training Qwen2-VL-2B with GRPO for robot navigation 
using rubric-based rewards.

## Installation

Install UV package manager:
curl -LsSf https://astral.sh/uv/install.sh | sh

Clone the repository:
git clone https://github.com/zhalehk/grpo-robot-navigation.git
cd grpo-robot-navigation

Install dependencies:
uv sync

## Requirements
- GPU with 16GB+ VRAM (tested on Kaggle T4)
- HuggingFace account with dataset access
- Weights & Biases account

## Environment Variables
Set these before running:
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
CEREBRAS_API_KEY=your_cerebras_key

## Training
uv run python train.py

## Evaluation
cd ~/reasoning-judge
python main.py --model gpt-oss-120b --output results.csv
