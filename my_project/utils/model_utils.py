import torch
import numpy as np
import pandas as pd
from scipy.stats import entropy
import einops
from fancy_einsum import einsum
from vit_prisma.utils import prisma_utils
from vit_prisma.utils.data_utils.race_dict import RACE_DICT
from utils.data_utils import get_image_from_batch

def process_corrupt_cache(corrupt_cache, amplifier):
    avg_corrupt_cache = {}
    for key, tensor in corrupt_cache.items():
        avg_tensor = tensor.mean(dim=0, keepdim=True)
        avg_tensor = avg_tensor * amplifier
        avg_corrupt_cache[key] = avg_tensor
    return avg_corrupt_cache

def patch_head_vector(original_head_vector, hook, head_index, cache):
    patch_num = 0
    original_head_vector[:, patch_num, head_index, :] = cache[hook.name][:, patch_num, head_index, :]
    return original_head_vector

def process_image(model, image, head_index, layer_number, corrupt_cache):
    logits = model.run_with_hooks(
        image.unsqueeze(0),
        fwd_hooks=[(prisma_utils.get_act_name('z', layer_number, "attn"), 
                    lambda original_head_vector, hook: patch_head_vector(original_head_vector, hook, head_index, corrupt_cache))]
    )
    return logits

def analyze_logits(logits, top_k=3):
    probs = logits.softmax(dim=-1)
    probs = probs.squeeze(0).detach().numpy()
    sorted_probs = np.sort(probs)[::-1]
    sorted_probs_args = np.argsort(probs)[::-1]

    results = []
    for i in range(top_k):
        index = sorted_probs_args[i]
        prob = sorted_probs[i]
        logit = logits[0, index].item()
        label = RACE_DICT[index]

        results.append({
            'rank': f"Top {i}th token.",
            'logit': f"Logit: {logit:.2f}",
            'prob': f"Prob: {prob * 100:.2f}%",
            'label': f"Label: |{label}|"
        })
    return results

def residual_stack_to_logit_attn(residual_stack, cache, answer_residual_direction) -> float:
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=0)
    average_scaled_residual_stack = scaled_residual_stack.mean(axis=1, keepdim=True)
    return einsum(
        "... batch d_model, batch d_model -> ...",
        scaled_residual_stack,
        answer_residual_direction,
    )

def get_target_layer_head(tensor_of_heads, model):
    for head in tensor_of_heads:
        layer = head // model.cfg.n_heads
        head_index = head % model.cfg.n_heads
        return layer.item(), head_index.item()

def compute_logit_differences(model, clean_output, clean_cache):
    per_head_residual, head_labels = clean_cache.stack_head_results(layer=-1, pos_slice=0, return_labels=True)

    logits = torch.randn(clean_output.shape[0], 3)  # Adjust the shape if necessary
    answer_tokens = torch.Tensor([[0, 1]]).long()
    answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)
    logit_diff_directions = answer_residual_directions[:, 0] - answer_residual_directions[:, 1]

    per_head_logit_diffs = residual_stack_to_logit_attn(per_head_residual, clean_cache, logit_diff_directions)
    per_head_logit_diffs = einops.rearrange(per_head_logit_diffs, "(layer head_index) -> layer head_index", layer=model.cfg.n_layers, head_index=model.cfg.n_heads)
    top_positive_logit_attr_heads = torch.topk(per_head_logit_diffs.flatten(), k=5).indices
    print('Head indices of positive logit attr tokens:', top_positive_logit_attr_heads)
    layer_number, head_index = get_target_layer_head(top_positive_logit_attr_heads, model)

    return layer_number, head_index

def get_probs(logits):
    probs = logits.softmax(dim=-1)
    return probs.squeeze(0)

def process_predictions(model, images, labels, range_size, head_index, layer_number, corrupt_cache):
    clean_logits_list = []
    corrupt_logits_list = []
    ground_truths = []

    for j in range(range_size):
        image = get_image_from_batch(images, labels, j)
        clean_logits = model(image.unsqueeze(0))
        corrupt_logits = model.run_with_hooks(
            image.unsqueeze(0),
            fwd_hooks=[(prisma_utils.get_act_name('z', layer_number, "attn"), 
                        lambda original_head_vector, hook: patch_head_vector(original_head_vector, hook, head_index, corrupt_cache))]
        )

        clean_logits_list.append(clean_logits)
        corrupt_logits_list.append(corrupt_logits)
        ground_truths.append(labels[j].item())

    clean_logits_tensor = torch.cat(clean_logits_list)
    corrupt_logits_tensor = torch.cat(corrupt_logits_list)
    ground_truths_tensor = torch.tensor(ground_truths)

    return clean_logits_tensor, corrupt_logits_tensor, ground_truths_tensor

def model_metrics(clean_logits, corrupt_logits, ground_truths, race_dict):
    reverse_race_dict = {v: k for k, v in race_dict.items()}

    clean_correct = 0
    corrupt_correct = 0
    
    clean_entropies = []
    corrupt_entropies = []
    total = clean_logits.size(0)
    
    for i in range(total):
        ground_truth = ground_truths[i]
        
        clean_probs = get_probs(clean_logits[i])
        corrupt_probs = get_probs(corrupt_logits[i])
        
        clean_pred_index = torch.argmax(clean_probs).item()
        corrupt_pred_index = torch.argmax(corrupt_probs).item()
        
        if clean_pred_index == ground_truth:
            clean_correct += 1
        if corrupt_pred_index == ground_truth:
            corrupt_correct += 1
        
        clean_entropies.append(entropy(clean_probs.detach().numpy(), base=2))
        corrupt_entropies.append(entropy(corrupt_probs.detach().numpy(), base=2))
    
    clean_accuracy = clean_correct / total
    corrupt_accuracy = corrupt_correct / total
    
    avg_clean_entropy = np.mean(clean_entropies)
    avg_corrupt_entropy = np.mean(corrupt_entropies)
    
    max_entropy = np.log2(3)
    
    clean_fairness = avg_clean_entropy / max_entropy
    corrupt_fairness = avg_corrupt_entropy / max_entropy
    
    return {
        'clean_accuracy': clean_accuracy,
        'corrupt_accuracy': corrupt_accuracy,
        'clean_fairness': clean_fairness,
        'corrupt_fairness': corrupt_fairness
    }
