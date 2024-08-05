import torch
import random
import numpy as np
from torchvision import transforms
from vit_prisma.utils.data_utils.race_dict import RACE_DICT

class ConvertTo3Channels:
    def __call__(self, img):
        if img.mode != 'RGB':
            return img.convert('RGB')
        return img

transform = transforms.Compose([
    ConvertTo3Channels(),  
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def transform_batch(examples):
    images = [transform(image) for image in examples['image']]
    label_map = {'WHITE': 0, 'BLACK/AFRICAN AMERICAN': 1, 'ASIAN': 2}  
    labels = [label_map[race] for race in examples['race']]
    labels = torch.tensor(labels, dtype=torch.long) 
    return {'image': images, 'label': labels}

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_image_from_batch(images, labels, idx):
    image = images[idx]
    label = labels[idx]
    label_str = RACE_DICT.get(label.item(), 'Unknown')
    return image
