import torch
from torch.utils.data import DataLoader
from datasets import DatasetDict
from utils.data_utils import transform_batch, worker_init_fn, get_image_from_batch, get_subset_loader
from utils.model_utils import process_corrupt_cache, process_image, analyze_logits, residual_stack_to_logit_attn, get_target_layer_head

def get_inference(model, batch_size, race=None, healthy=True, test=True):
    if healthy:
        dataset = DatasetDict.load_from_disk("/content/drive/MyDrive/all_healthy")
        if test:
            dataset = dataset["test"]
        else:
            dataset = dataset["validate"]
    else: 
        dataset = DatasetDict.load_from_disk("/content/drive/MyDrive/sampled_diseased")["test"]
    
    loader = get_subset_loader(batch_size,dataset,race=race)
    data = next(iter(loader))
    images, labels = data['image'], data['label']
    output, cache = model.run_with_cache(images)
    
    return images, labels, output, cache

def process_and_analyze(model, images, labels, head_index, layer_number, batch_size, corrupt_cache):
    for j in range(batch_size):
        image = get_image_from_batch(images, labels, j)
        logits = process_image(model, image, head_index, layer_number, corrupt_cache)
        results = analyze_logits(logits)

        for result in results:
            print(f"{result['rank']} {result['logit']} {result['prob']} {result['label']}")
