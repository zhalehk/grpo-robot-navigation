import io
import logging
from PIL import Image
from datasets import load_dataset

from config import DATASET_NAME, DATASET_SPLIT

logger = logging.getLogger(__name__)


def get_pil_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    elif isinstance(image, dict):
        if "bytes" in image and image["bytes"]:
            return Image.open(
                io.BytesIO(image["bytes"])
            ).convert("RGB")
        elif "path" in image and image["path"]:
            return Image.open(image["path"]).convert("RGB")
    return None


def format_single_example(example):
    task  = example["task"]
    image = get_pil_image(example["image"])
    messages = [
        {
            "role": "system",
            "content": "You are a robot navigation assistant. You MUST start your response immediately with <motivation> — no text before it. Format: <motivation>reasoning</motivation><score>0, 1, or 2</score>. Nothing before <motivation>, nothing after </score>."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": task},
            ],
        }
    ]
    return {
        "prompt":    messages,
        "reasoning": example["reasoning"],
        "score":     int(example["score"]),
    }

class RobotDataset:
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return format_single_example(self.dataset[idx])  # PIL created fresh each time ✅




def load_robot_dataset(dataset_name=DATASET_NAME, dataset_split=DATASET_SPLIT):
    logger.info(f"Loading {dataset_name} ({dataset_split})...")

    raw_dataset = load_dataset(
        dataset_name,
        split=dataset_split,
    )

    logger.info(f"Dataset loaded! Rows: {len(raw_dataset)}")

    # Filter invalid rows
    def is_valid(example):
        score = example.get("score")
        task  = example.get("task", "")
        return score in [0, 1, 2] and bool(task)

    filtered = raw_dataset.filter(is_valid)
    logger.info(f"Valid rows: {len(filtered)}")

    dataset = RobotDataset(filtered)
    logger.info(f"Dataset ready! {len(dataset)} examples")
    # DEBUG
    sample = dataset[0]
    print("\n=== FULL PROMPT ===")
    for msg in sample["prompt"]:
        content = msg["content"]
        if isinstance(content, str):
            print(content)
        elif isinstance(content, list):
            for part in content:
                if part["type"] == "text":
                    print(part["text"])
    print("=== END PROMPT ===\n")
    return dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing dataset loading...")
    dataset = load_robot_dataset()

    print(f"\nDataset size: {len(dataset)}")
    example = dataset[0]
    print(f"Score: {example['score']}")
    print(f"Reasoning: {example['reasoning'][:100]}...")
    print(f"Prompt type: {type(example['prompt'])}")
    print("\nDone! ✅")
