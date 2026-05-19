# GRPO Training for SmolVLM on Robot Navigation

Complete training implementation for SmolVLM-256M-Instruct using GRPO on a robot navigation dataset.

## 📋 Project Overview

**Goal:** Train a Vision Language Model to generate reasoning about robot navigation images and assign relevance scores.

**Model Output Format:**
```xml
<motivation>
I can see a white marble desk with a polished surface.
The texture is smooth and reflective.
The desk is positioned in the left corner of the room.
</motivation>
<score>2</score>
```

**Dataset:** reasoning-augmentation/reasoning_distractors_choice_deduped (9,975 examples)

**Model:** HuggingFaceTB/SmolVLM-256M-Instruct (256M parameters, optimized for 8GB GPU)

---

## 📁 File Structure

| File | Purpose |
|------|---------|
| `config.py` | Centralized configuration: model names, hyperparameters, API settings |
| `cerebras_client.py` | Cerebras API client for evaluating reasoning quality (3 axes) |
| `dataset.py` | Dataset loading and formatting for training |
| `reward.py` | 3-part reward function (template + score + reasoning) |
| `train.py` | Main training orchestrator |
| `README.md` | This file |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
source ~/grpo_env/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets trl wandb accelerate requests python-dotenv
pip install vllm
```

### 2. Set Up API Keys

Create `~/.env.ml` with your credentials:

```bash
cat > ~/.env.ml << 'EOF'
HF_TOKEN=hf_your_token_here
CEREBRAS_API_KEY=csk_your_key_here
EOF

chmod 600 ~/.env.ml
```

Get API keys from:
- **HuggingFace Token:** https://huggingface.co/settings/tokens
- **Cerebras API Key:** https://cloud.cerebras.ai/console

### 3. Run Training

```bash
cd ~/Desktop/grpo_robot_navigation

# Start training
FLASHINFER_DISABLE_VERSION_CHECK=1 accelerate launch train.py
```

---

## 📊 Configuration Reference

Edit `config.py` to customize:

### Model & Dataset
```python
MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"
DATASET_NAME = "reasoning-augmentation/reasoning_distractors_choice_deduped"
DATASET_SPLIT = "train"
OUTPUT_DIR = "./smolvlm_grpo_robot"
```

### Reward Weights (sum to 1.0)
```python
ALPHA = 0.5    # Reasoning quality weight
BETA = 0.3     # Score correctness weight
GAMMA = 0.2    # Template compliance weight
```

### Training Hyperparameters
```python
NUM_EPOCHS = 2
BATCH_SIZE = 2
LEARNING_RATE = 1e-6
MAX_COMPLETION_LENGTH = 256
NUM_GENERATIONS = 2
VLLM_GPU_MEMORY = 0.3
```

### Cerebras API
```python
CEREBRAS_MODEL = "llama3.1-8b"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
CEREBRAS_MAX_RETRIES = 3
CEREBRAS_RETRY_DELAY = 1
```

---

## 🏗️ How It Works

### Data Flow

```
Dataset (HuggingFace)
    ↓
dataset.py: Load & Format
    ↓
[IMAGE] + task → prompt
Preserve: score, reasoning
    ↓
train.py: GRPOTrainer
    ↓
Generate: <motivation>...</motivation>
          <score>X</score>
    ↓
reward.py: Compute 3-part reward
```

### Reward Function

The training uses a **3-part reward function**:

#### 1. Template Reward (γ=0.2)
Checks proper formatting:
```python
+0.5 if has <motivation> tags
+0.5 if has <score> tags
```

#### 2. Score Reward (β=0.3)
Compares predicted vs ground truth score:
```python
+1.0 if exact match (0→0, 1→1, 2→2)
+0.5 if off by 1
+0.0 if off by 2+
```

#### 3. Reasoning Reward (α=0.5)
Uses **Cerebras API** to evaluate on 3 axes:
1. **Color Accuracy (0-5):** Are colors correctly identified?
2. **Texture/Material (0-5):** Are materials accurately described?
3. **Spatial/Context (0-5):** Are spatial relationships correct?

Each axis normalized: 0-5 → 0.0-1.0, then averaged.

#### Final Calculation
```
final_reward = 0.5 * reasoning + 0.3 * score + 0.2 * template
             = [0.0, 1.0]
```

---

## 📈 Training Output

### Console Logs

```
======================================================================
GRPO Training: SmolVLM for Robot Navigation
======================================================================

[1/6] Validating environment...
✓ Environment validated

[2/6] Loading dataset...
✓ Loaded: 9975 examples

[3/6] Loading model...
✓ Model loaded: 256,000,000 parameters

[4/6] Setting up trainer...
✓ Trainer initialized

[5/6] Verifying APIs...
✓ Cerebras API healthy

[6/6] Starting training...
  Epochs: 2
  Batch size: 2
  Learning rate: 1e-6

[Training progress...]

✓ Training completed
✓ Saved to ./smolvlm_grpo_robot

======================================================================
Pipeline: 7/7 steps completed
✓ All steps completed successfully!
======================================================================
```

### Weights & Biases Logging

Metrics logged in real-time to the "smolvlm-grpo-training" project:
- `train/loss` - GRPO loss per step
- `train/reward` - Mean reward per batch
- `train/learning_rate` - Learning rate schedule
- `train/policy_logps` - Policy log probabilities
- `train/entropy` - Policy entropy

### Saved Model

After training, `./smolvlm_grpo_robot/` contains:
```
smolvlm_grpo_robot/
├── pytorch_model.bin           # Trained weights
├── preprocessor_config.json    # Image processor config
├── config.json                 # Model config
├── training_args.bin           # Training arguments
└── checkpoint-*/               # Periodic checkpoints
```

---

## 🧪 Testing Individual Modules

Each module has standalone testing:

### Test Dataset Loading
```bash
python dataset.py
```
Output:
```
[1/2] Loading dataset...
✓ Dataset loaded: 9975 examples

[2/2] Inspecting first example...
Prompt: [IMAGE] Navigate to the white marble desk
Task: Navigate to the white marble desk
Ground truth score: 2
Ground truth reasoning: There is a white marble desk...
```

### Test Reward Function
```bash
python reward.py
```
Output:
```
=== Testing Reward Function ===

[1/3] Test perfect completion...
  Template: 1.000, Score: 1.000

[2/3] Test missing tags...
  Template: 0.000, Score: 0.000

[3/3] Test off-by-one score...
  Template: 1.000, Score: 0.500

✓ Reward function tests complete
```

### Test Cerebras API
```bash
python cerebras_client.py
```
Output:
```
[1/2] Running health check...
✓ Cerebras API health check passed

[2/2] Testing reasoning evaluation...
Evaluation result: 0.867
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory

**Error:** `CUDA out of memory`

**Solution:**
```python
# In config.py, reduce:
BATCH_SIZE = 1                  # was 2
NUM_GENERATIONS = 1             # was 2
MAX_COMPLETION_LENGTH = 128     # was 256
VLLM_GPU_MEMORY = 0.2          # was 0.3
```

### Dataset Not Found

**Error:** `OSError: reasoning-auguration not found`

**Solution:**
- Verify HF_TOKEN is set in `~/.env.ml`
- Check HuggingFace Hub is accessible
- Verify internet connection

### Cerebras API Unreachable

**Error:** `Failed to connect to Cerebras API`

**Solution:**
- Verify CEREBRAS_API_KEY in `~/.env.ml`
- Check API key is valid and has credits
- Training continues with fallback scores (0.5)

### vLLM Not Found

**Error:** `ModuleNotFoundError: No module named 'vllm'`

**Solution:**
```bash
pip install vllm

# Or disable in config.py:
USE_VLLM = False
```

### FLASHINFER Version Check

**Error:** Environment variable not recognized

**Solution:**
Always run with:
```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 accelerate launch train.py
```

---

## 📚 Key Concepts

### GRPO
Generative Reward Preference Optimization - a policy gradient algorithm that:
1. Generates multiple completions per prompt
2. Computes rewards for each completion
3. Updates policy to maximize expected reward

### Cerebras API
Lightweight LLM API for evaluating reasoning quality. Called once per batch during training to compare model reasoning with ground truth on semantic axes.

### vLLM Backend
Efficient generation engine for parallel sampling. Used by GRPOTrainer to generate multiple completions per prompt.

---

## 📊 Performance Expectations

**Hardware:** NVIDIA RTX 4060 8GB GPU

- **Training time per epoch:** 30-40 minutes
- **Total (2 epochs):** 60-80 minutes
- **GPU memory:** 6.5-7.0 GB peak
- **Model size:** ~600 MB

**Tips for faster training:**
1. Increase `LOGGING_STEPS` from 10 to 100
2. Disable W&B logging: `REPORT_TO = []`
3. Use `USE_VLLM = False` if API is slow
4. Reduce `CEREBRAS_TIMEOUT` if API calls timeout

---

## 🔄 After Training

### Evaluate Model

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

processor = AutoProcessor.from_pretrained("./smolvlm_grpo_robot")
model = AutoModelForVision2Seq.from_pretrained("./smolvlm_grpo_robot")

# Load image and generate
image = Image.open("robot_image.jpg")
prompt = "[IMAGE] Navigate to the white marble desk"

inputs = processor(image, prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=256)
response = processor.decode(output[0], skip_special_tokens=True)

print(response)
```

### Push to HuggingFace Hub

```bash
huggingface-cli repo create smolvlm-grpo-robot-navigation

python -c "
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained('./smolvlm_grpo_robot')
model = AutoModelForVision2Seq.from_pretrained('./smolvlm_grpo_robot')

model.push_to_hub('your-username/smolvlm-grpo-robot-navigation')
processor.push_to_hub('your-username/smolvlm-grpo-robot-navigation')
"
```

---

## 📞 Support

- Check logs for detailed error messages
- Review configuration in `config.py`
- Test individual modules with their `__main__` sections
- Enable debug logging by modifying `logging.basicConfig(level=logging.DEBUG)`

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@misc{grpo_smolvlm_robot_nav,
  title={GRPO Training for Robot Navigation with SmolVLM},
  author={Your Name},
  year={2025}
}
```

---

## 📝 License

This project is for academic research purposes.
