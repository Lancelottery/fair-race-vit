import sys
import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

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
    # white_images, white_labels, white_output, white_cache = get_inference(model, 16, "WHITE")
    # black_images, black_labels, black_output, black_cache = get_inference(model, 16, "BLACK/AFRICAN AMERICAN")

    healthy_test_images, healthy_test_labels, healthy_test_output, healthy_test_cache = get_inference(model, 100, race=None, healthy=True, test=True)
    healthy_val_images, healthy_val_labels, healthy_val_output, healthy_val_cache = get_inference(model, 100, race=None, healthy=True, test=False)
    # diseased_images, diseased_labels, diseased_output, diseased_cache = get_inference(model, 100, race=None, healthy=False)

    # Compute logit differences
    # layer_number, head_index = compute_logit_differences(model, white_output, white_cache)

    # Initialize variables
    amplifier = 32
    results = []

    # Process corrupted cache
    avg_corrupt_cache = process_corrupt_cache(healthy_val_cache, amplifier)

    # Compute clean model metrics once
    clean_logits, corrupt_logits, ground_truths = process_predictions(model, healthy_test_images, healthy_test_labels, 100, 0, 0, avg_corrupt_cache)
    metrics = model_metrics(clean_logits, corrupt_logits, ground_truths, RACE_DICT)

    clean_accuracy = metrics['clean_accuracy']
    clean_fairness = metrics['clean_fairness']

    # Print clean model metrics
    print(f"Clean Model Accuracy: {clean_accuracy:.2f}")
    print(f"Clean Model Fairness (0 to 1): {clean_fairness}")

    # Nested loop for layer_number and head_index with tqdm progress tracking
    total_iterations = 12 * 12
    for layer_number, head_index in tqdm(product(range(12), repeat=2), total=total_iterations, desc="Total Progress"):
        # Process predictions for current layer_number and head_index
        clean_logits, corrupt_logits, ground_truths = process_predictions(model, healthy_test_images, healthy_test_labels, 100, head_index, layer_number, avg_corrupt_cache)
        
        # Calculate metrics
        metrics = model_metrics(clean_logits, corrupt_logits, ground_truths, RACE_DICT)
        
        # Store results
        results.append({
            'layer_number': layer_number,
            'head_index': head_index,
            'corrupt_accuracy': metrics['corrupt_accuracy'],
            'corrupt_fairness': metrics['corrupt_fairness']
        })


    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV
    results_df.to_csv('amp32_change_head.csv', index=False)


if __name__ == "__main__":
    main()
