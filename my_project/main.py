import sys
import os

# Append the path to vit_prisma
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import torch
import einops
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.utils.data_utils.race_dict import RACE_DICT
from inference.inference import get_inference
from utils.data_utils import set_seed, get_image_from_batch
from utils.model_utils import residual_stack_to_logit_attn, process_corrupt_cache, compute_logit_differences, process_predictions, model_metrics

def main():
    set_seed(42)

    # Load model
    model = torch.load('../weight/healthy32-ac78/healthy_model.pth')

    # Get inferences
    white_images, white_labels, white_output, white_cache = get_inference(model, "healthy_test_white", 16)
    black_images, black_labels, black_output, black_cache = get_inference(model, "healthy_val_white", 16)

    # Compute logit differences
    layer_number, head_index = compute_logit_differences(model, white_output, white_cache)

    # Process corrupted cache
    amplifier = 20
    avg_corrupt_cache = process_corrupt_cache(black_cache, amplifier)

    # Process predictions
    clean_logits, corrupt_logits, ground_truths = process_predictions(model, white_images, white_labels, 16, head_index, layer_number, avg_corrupt_cache)

    # Calculate metrics
    metrics = model_metrics(clean_logits, corrupt_logits, ground_truths, RACE_DICT)

    print(f"Clean Model Accuracy: {metrics['clean_accuracy']:.2f}")
    print(f"Corrupt Model Accuracy: {metrics['corrupt_accuracy']:.2f}")
    print(f"Clean Model Fairness (0 to 1): {metrics['clean_fairness']}")
    print(f"Corrupt Model Fairness (0 to 1): {metrics['corrupt_fairness']}")

if __name__ == "__main__":
    main()
