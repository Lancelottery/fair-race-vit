import torch
from torch.utils.data import DataLoader
from datasets import DatasetDict
from utils.data_utils import transform_batch, worker_init_fn, get_image_from_batch
from utils.model_utils import process_corrupt_cache, process_image, analyze_logits, residual_stack_to_logit_attn, get_target_layer_head

def get_inference(model, file, batch_size):
    path = f"/content/drive/MyDrive/{file}"
    dataset = DatasetDict.load_from_disk(path)
    dataset.set_transform(transform_batch)
    
    loader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)
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
