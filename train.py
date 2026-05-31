import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import logging
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from trl import GRPOTrainer, GRPOConfig
from config import (
    MODEL_NAME, OUTPUT_DIR, DATASET_NAME,
    DATASET_SPLIT, NUM_EPOCHS, BATCH_SIZE, PER_DEVICE_TRAIN_BATCH_SIZE,
    LEARNING_RATE, MAX_COMPLETION_LENGTH,
    NUM_GENERATIONS, SAVE_STEPS, LOGGING_STEPS
)
from peft import LoraConfig, get_peft_model
from dataset import load_robot_dataset
from reward import reward_function

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("  SmolVLM GRPO Training - Robot Navigation")
    logger.info("=" * 60)
    logger.info(f"  Model:   {MODEL_NAME}")
    logger.info(f"  Output:  {OUTPUT_DIR}")
    logger.info("=" * 60)
    logger.info("")

    logger.info("[1/5] Validating environment...")
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found!")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name}")
    logger.info(f"GPU Memory: {gpu_mem:.1f} GB")
    logger.info("Environment OK!")
    logger.info("")

    logger.info("[2/5] Loading dataset...")
    train_dataset = load_robot_dataset(DATASET_NAME, DATASET_SPLIT)
    logger.info(f"Dataset loaded! {len(train_dataset)} examples")
    logger.info("")

    logger.info("[3/5] Loading processor...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    logger.info("Processor loaded!")
    logger.info("")

    logger.info("[4/5] Setting up GRPO trainer...")
    try:
        config = GRPOConfig(
            output_dir                   = OUTPUT_DIR,
            num_train_epochs             = NUM_EPOCHS,
            per_device_train_batch_size  = PER_DEVICE_TRAIN_BATCH_SIZE,
            num_generations              = NUM_GENERATIONS,
            max_completion_length        = MAX_COMPLETION_LENGTH,
            learning_rate                = LEARNING_RATE,
            logging_steps                = LOGGING_STEPS,
            save_steps                   = SAVE_STEPS,
            save_total_limit             = 1,
            bf16                         = True,
            fp16                         = False,
            gradient_checkpointing       = True,
            use_vllm                     = False,
            report_to                    = "wandb",
        )
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=128,
            lora_alpha=64,
            target_modules="all-linear",
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
            #attn_implementation="flash_attention_2",
        )

        model = get_peft_model(model, lora_config)

        print(model.print_trainable_parameters())
        trainer = GRPOTrainer(
            model            = model,
            args             = config,
            train_dataset    = train_dataset,
            processing_class = processor,
            reward_funcs     = reward_function,
        )
        logger.info("Trainer ready!")
    except Exception as e:
        logger.error(f"Trainer setup failed: {e}")
        raise

    logger.info("[5/5] Starting training...")
    logger.info(f"  Epochs:       {NUM_EPOCHS}")
    logger.info(f"  Batch size:   {BATCH_SIZE}")
    logger.info(f"  LR:           {LEARNING_RATE}")
    logger.info(f"  Generations:  {NUM_GENERATIONS}")
    logger.info("")

    try:
        trainer.train()
        logger.info("Training complete! ✅")
        trainer.save_model(OUTPUT_DIR)
        logger.info(f"Model saved to {OUTPUT_DIR}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
