import re
import logging
from config import ALPHA, BETA, GAMMA

logger = logging.getLogger(__name__)

COLOR_WORDS = [
    "white", "black", "red", "blue", "green",
    "yellow", "brown", "gray", "grey", "orange",
    "purple", "pink", "dark", "light", "bright",
    "coral", "beige", "cream", "golden", "silver",
    "teal", "navy", "wooden", "marble",
]

TEXTURE_WORDS = [
    "wood", "wooden", "marble", "glass", "metal",
    "plastic", "fabric", "leather", "smooth", "rough",
    "polished", "matte", "glossy", "grain", "laminate",
    "painted", "natural", "rustic", "concrete", "brick",
    "velvet", "steel", "chrome", "ceramic", "linen",
]

SPATIAL_WORDS = [
    "left", "right", "center", "middle", "corner",
    "against", "beside", "near", "next", "above",
    "below", "front", "back", "wall", "floor",
    "window", "positioned", "placed", "adjacent",
    "behind", "facing", "side", "top", "bottom",
]

def extract_motivation_and_score(text):
    motivation = ""
    match = re.search(r"<motivation>(.*?)</motivation>", text, re.DOTALL | re.IGNORECASE)
    if match:
        motivation = match.group(1).strip()
    score = None
    score_match = re.search(r"<score>([012])</score>", text, re.IGNORECASE)
    if score_match:
        try:
            score = int(score_match.group(1))
        except ValueError:
            score = None
    return motivation, score

def compute_template_reward(text):
    reward = 0.0
    has_motivation = bool(re.search(r"<motivation>.*?</motivation>", text, re.DOTALL | re.IGNORECASE))
    has_score = bool(re.search(r"<score>[012]</score>", text, re.IGNORECASE))
    if has_motivation:
        reward += 0.5
    if has_score:
        reward += 0.5
    return reward

def compute_score_reward(predicted, ground_truth):
    if predicted is None:
        return 0.0
    diff = abs(predicted - int(ground_truth))
    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.5
    return 0.0

def compute_reasoning_reward(motivation, gt_reasoning):
    if not motivation or not gt_reasoning:
        return 0.0
    m = motivation.lower()
    g = gt_reasoning.lower()
    gt_colors = set(w for w in COLOR_WORDS if w in g)
    pred_colors = set(w for w in COLOR_WORDS if w in m)
    color = len(pred_colors & gt_colors) / len(gt_colors) if gt_colors else 0.5
    gt_tex = set(w for w in TEXTURE_WORDS if w in g)
    pred_tex = set(w for w in TEXTURE_WORDS if w in m)
    texture = len(pred_tex & gt_tex) / len(gt_tex) if gt_tex else 0.5
    gt_spa = set(w for w in SPATIAL_WORDS if w in g)
    pred_spa = set(w for w in SPATIAL_WORDS if w in m)
    spatial = len(pred_spa & gt_spa) / len(gt_spa) if gt_spa else 0.5
    reasoning_r = (color + texture + spatial) / 3.0
    print(f"    color={color:.2f} texture={texture:.2f} spatial={spatial:.2f} → reasoning={reasoning_r:.2f}")
    return reasoning_r

def reward_function(prompts, completions, reasoning, score, **kwargs):
    rewards = []
    for completion, gt_reasoning, gt_score in zip(completions, reasoning, score):
        if isinstance(completion, list):
            text = completion[0]["content"]
        else:
            text = str(completion)
        motivation, pred_score = extract_motivation_and_score(text)
        template_r = compute_template_reward(text)
        score_r = compute_score_reward(pred_score, gt_score)
        reasoning_r = 0.0
        if motivation:
            reasoning_r = compute_reasoning_reward(motivation, gt_reasoning)
        final = ALPHA * reasoning_r + BETA * score_r + GAMMA * template_r
        final = max(0.0, min(1.0, final))
        rewards.append(final)
        print(f"  template={template_r:.2f} score={score_r:.2f} reasoning={reasoning_r:.2f} → final={final:.2f}")
    return rewards
