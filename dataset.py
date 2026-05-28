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
    task = example["task"]  # already has full prompt!
    reasoning = example["reasoning"]
    score     = int(example["score"])
    image     = get_pil_image(example["image"])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": task},  # use task directly!
            ],
        }
    ]
    return {
        "prompt":    messages,
        "reasoning": reasoning,
        "score":     score,
    }
    




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

    dataset = filtered.map(format_single_example, remove_columns=filtered.column_names)
    logger.info(f"Dataset ready! {len(dataset)} examples")
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
