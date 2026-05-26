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
    task      = example["task"]
    reasoning = example["reasoning"]
    score     = int(example["score"])
    image     = get_pil_image(example["image"])

    instruction = f"""You are a robot navigating an enclosed space.
Your goal is to navigate to the correct object based on the
user's commands. You were given the following task by the
user '{task}'. Currently, you are facing a scene represented
by the given image. Reason about what you are seeing, comparing
what you know about the task (given the user commands) and the
given scene. For example, if the task is 'Navigate to the black
leather sofa near a lampstand' your reasoning process will be
'I'm currently observing a brown sofa which is different than
black, making it unlikely to be the target sofa. Moreover, there
is no lampstand near it, only a rug and a window' etc. If there
are distortions or artifact, do not focus on them, focus on the
object at hand. At the end of the reasoning process, evaluate
how well the provided image aligns with the user's task. Assign
a confidence score based on the following scale: - 0: You are
certain the image DOES NOT match the task. - 1: You are unsure
whether the image matches the task or not. - 2: You are certain
the image DOES match the task. Strictly start your response with <motivation> with no text before it, and follow this exact format:
<motivation>Your reasoning here</motivation><score>0, 1, or 2</score>"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": instruction,
                },
            ],
        }
    ]

    return {
        "prompt":    messages,
        "reasoning": reasoning,
        "score":     score,
    }


class RobotDataset:
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        return format_single_example(example)


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
