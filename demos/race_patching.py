import numpy as np
import random
import torch
import einops
from fancy_einsum import einsum
from collections import defaultdict
from torchvision import transforms
from torch.utils.data import DataLoader
import datasets
from datasets import load_dataset, DatasetDict
from IPython.display import display, HTML

import sys
sys.path.append('../src')

from vit_prisma.models.base_vit import HookedViT
from vit_prisma.utils.data_utils.race_utils import race_index_from_word
from vit_prisma.utils.data_utils.race_dict import RACE_DICT
from vit_prisma.utils import prisma_utils
from vit_prisma.utils.prisma_utils import test_prompt
from vit_prisma.prisma_tools.race_logit_lens import get_patch_logit_directions, get_patch_logit_dictionary

# n_layers: 12, d_model: 768, d_head: 64, 
# n_heads: 12 (per layer), d_mlp: 3072, patch_size: 32
model = torch.load('../weight/healthy32-ac78/healthy_model.pth')


class ConvertTo3Channels:
    def __call__(self, img):
        if img.mode != 'RGB':
            return img.convert('RGB')
        return img

transform = transforms.Compose([
    ConvertTo3Channels(),  # Ensure all images are 3-channel without turning them grayscale
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def transform_batch(examples):
    images = [transform(image) for image in examples['image']]
    label_map = {'WHITE': 0, 'BLACK/AFRICAN AMERICAN': 1, 'ASIAN':2}  
    labels = [label_map[race] for race in examples['race']]
    labels = torch.tensor(labels, dtype=torch.long) 

    return {'image': images, 'label': labels}

def get_attn_across_datapoints(attn_head_idx, attention_type="attn_scores"):

  list_of_attn_heads = [attn_head_idx]

  # Retrieve the activations from each batch idx
  all_patterns = []
  for batch_idx in range(images.shape[0]):
    patterns = visualize_attention(list_of_attn_heads, cache, "Attention Scores", 700, batch_idx = batch_idx, attention_type=attention_type)
    all_patterns.extend(patterns)
  all_patterns = torch.stack(all_patterns, dim=0)
  return all_patterns

# Set global seeds
def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    np.random.seed(seed_value)  # Numpy module.
    random.seed(seed_value)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_value = 42
set_seed(seed_value)

# Define a worker init function that sets the seed
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_image_from_batch(images,labels,idx):
    image = images[idx]
    label = labels[idx]
    label_str = RACE_DICT.get(label.item(), 'Unknown')
    print("Ground truth: ", label_str)
    return image
    
def get_inference(file, batch_size):
    path = f"/content/drive/MyDrive/{file}"
    dataset = DatasetDict.load_from_disk(path)
    dataset.set_transform(transform_batch)
    
    loader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)
    data = next(iter(loader))
    images, labels = data['image'], data['label']
    
    output, cache = model.run_with_cache(images)
    
    return images, labels, output, cache


# diseased_dataset_black/  healthy_test_asian/  healthy_val_all/    healthy_val_white/  hf_dataset_asian/
# diseased_dataset_all/    diseased_dataset_white/  healthy_test_black/  healthy_val_asian/  hf_dataset/         hf_dataset_black/
# diseased_dataset_asian/  healthy_test_all/        healthy_test_white/  healthy_val_black/  hf_dataset_all/     hf_dataset_white/

white_images, white_labels, white_output, white_cache = get_inference("healthy_test_white",16)
black_images, black_labels, black_output, black_cache = get_inference("healthy_test_black",16)
# asian_images, asian_labels, asian_output, asian_cache = get_inference("healthy_test_asian",3)

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


def residual_stack_to_logit_attn(residual_stack,cache,answer_residual_direction) -> float:
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=0)
    # average across batch!!
    average_scaled_residual_stack = scaled_residual_stack.mean(axis=1,keepdim=True)
    print(scaled_residual_stack.shape)
    print(answer_residual_direction.shape)
    return einsum(
        "... batch d_model, batch d_model -> ...",
        scaled_residual_stack,
        answer_residual_direction,
    )

output, batch_cache = white_output, white_cache
per_head_residual, head_labels = batch_cache.stack_head_results(
    layer=-1, pos_slice=0, return_labels=True
)

white_index = 0
black_index = 1
logits = torch.randn(16, 3)  # Batch of 16 data points
answer_tokens = [[white_index, black_index]]
answer_tokens = torch.Tensor(answer_tokens).long()
answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)
logit_diff_directions = answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
print("Logit difference directions shape:", logit_diff_directions.shape)

per_head_logit_diffs = residual_stack_to_logit_attn(per_head_residual, batch_cache, logit_diff_directions)

per_head_logit_diffs = einops.rearrange(
    per_head_logit_diffs,
    "(layer head_index) -> layer head_index",
    layer=model.cfg.n_layers,
    head_index=model.cfg.n_heads,
)

top_k = 1
top_positive_logit_attr_heads = torch.topk(
    per_head_logit_diffs.flatten(), k=top_k
).indices

def get_target_layer_head(tensor_of_heads):
  for head in tensor_of_heads:
    layer = head // model.cfg.n_heads
    head_index = head % model.cfg.n_heads
    return layer.item(), head_index.item()


def main(model, white_images, white_labels, head_index, layer_number, batch_size, corrupt_cache):
    for j in range(batch_size):
        image = get_image_from_batch(white_images, white_labels, j)
        logits = process_image(model, image, head_index, layer_number, corrupt_cache)
        results = analyze_logits(logits)

        for result in results:
            print(f"{result['rank']} {result['logit']} {result['prob']} {result['label']}")


layer_number, head_index = get_target_layer_head(top_positive_logit_attr_heads)
batch_size = 3
amplifier = 20
avg_corrupt_cache = process_corrupt_cache(black_cache, amplifier)

main(model, white_images, white_labels, head_index, layer_number, batch_size, avg_corrupt_cache)



# test_prompt(images[batch_idx],model)