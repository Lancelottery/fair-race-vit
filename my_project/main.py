import sys
import os
import csv
from tqdm import tqdm

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
    layer_number, head_index = compute_logit_differences(model, healthy_test_output, healthy_test_cache)

    print(layer_number, head_index)


    # # Process corrupted cache
    # amplifier = 20
    # # avg_corrupt_cache = process_corrupt_cache(black_cache, amplifier)
    # avg_corrupt_cache = process_corrupt_cache(healthy_val_cache, amplifier)
    # # Process predictions
    # # clean_logits, corrupt_logits, ground_truths = process_predictions(model, white_images, white_labels, 16, head_index, layer_number, avg_corrupt_cache)
    # clean_logits, corrupt_logits, ground_truths = process_predictions(model, healthy_test_images, healthy_test_labels, 64, head_index, layer_number, avg_corrupt_cache)

    # # Calculate metrics
    # metrics = model_metrics(clean_logits, corrupt_logits, ground_truths, RACE_DICT)

    # print(f"Clean Model Accuracy: {metrics['clean_accuracy']:.2f}")
    # print(f"Corrupt Model Accuracy: {metrics['corrupt_accuracy']:.2f}")
    # print(f"Clean Model Fairness (0 to 1): {metrics['clean_fairness']}")
    # print(f"Corrupt Model Fairness (0 to 1): {metrics['corrupt_fairness']}")

    # amplifier_values = [2**i for i in range(1, 9)]

    # # Prepare the CSV file
    # csv_file = "results.csv"
    # with open(csv_file, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Amplifier", "Clean Accuracy", "Clean Fairness", "Corrupt Accuracy", "Corrupt Fairness"])

    # # Store the results
    # results = []

    # for amplifier in tqdm(amplifier_values, desc="Processing amplifiers"):
    #     # Process corrupted cache
    #     avg_corrupt_cache = process_corrupt_cache(healthy_val_cache, amplifier)
        
    #     # Process predictions
    #     clean_logits, corrupt_logits, ground_truths = process_predictions(
    #         model, healthy_test_images, healthy_test_labels, 100, head_index, layer_number, avg_corrupt_cache
    #     )

    #     # Calculate metrics
    #     metrics = model_metrics(clean_logits, corrupt_logits, ground_truths, RACE_DICT)

    #     clean_accuracy = metrics['clean_accuracy']
    #     corrupt_accuracy = metrics['corrupt_accuracy']
    #     clean_fairness = metrics['clean_fairness']
    #     corrupt_fairness = metrics['corrupt_fairness']

    #     # Append the result to the list
    #     results.append((amplifier, clean_accuracy, clean_fairness, corrupt_accuracy, corrupt_fairness))
        
    #     # Write the result to the CSV file
    #     with open(csv_file, mode='a', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow([amplifier, clean_accuracy, clean_fairness, corrupt_accuracy, corrupt_fairness])

    #     print(f"Amplifier: {amplifier}")
    #     print(f"Clean Model Accuracy: {clean_accuracy:.2f}")
    #     print(f"Corrupt Model Accuracy: {corrupt_accuracy:.2f}")
    #     print(f"Clean Model Fairness (0 to 1): {clean_fairness}")
    #     print(f"Corrupt Model Fairness (0 to 1): {corrupt_fairness}")

    # # Identify the best amplifier value(s)
    # best_results = []
    # best_fairness = max(results, key=lambda x: x[4])[4]  # Highest corrupt fairness
    # lowest_accuracy = min(results, key=lambda x: x[3])[3]  # Lowest corrupt accuracy

    # for result in results:
    #     if result[3] == lowest_accuracy and result[4] == best_fairness:
    #         best_results.append(result[0])

    # print(f"The best amplifier values are: {best_results}")

if __name__ == "__main__":
    main()
