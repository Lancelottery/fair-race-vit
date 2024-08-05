import numpy as np
import string, random, json
import torch
import os
from typing import List
from jinja2 import Template
import ipywidgets as widgets
import matplotlib.pyplot as plt

def convert_to_3_channels(image):
    # Check if the image has only one channel (grayscale)
    if image.shape[-1] == 1 or image.ndim == 2:
        # Stack the grayscale image three times along the third axis to make it 3-channel
        image = np.squeeze(image)
        image = np.stack([image, image, image], axis=-1)
    return image

def generate_random_string(length=10):
    '''
    Helper function to generate canvas IDs for javascript figures.
    '''
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def prepare_image(image):
    if isinstance(image,torch.Tensor):
        image = image.numpy()
    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = image.astype('uint8')
    image = np.transpose(image, (1, 2, 0))
    image = convert_to_3_channels(image)
    return image

def flatten_into_patches(image, patch_size, image_size):
    patches = [image[i:i+patch_size, j:j+patch_size, :] for i in range(0, image_size, patch_size) for j in range(0, image_size, patch_size)]
    flattened_patches = [patch.flatten().tolist() for patch in patches]
    return flattened_patches

def normalize_attn_head(attn_head):
    if isinstance(attn_head, torch.Tensor):
        attn_head = attn_head.detach().cpu().numpy()
    min_val = np.min(attn_head)
    max_val = np.max(attn_head)
    normalized_attn_head = (attn_head - min_val) / (max_val - min_val)
    return normalized_attn_head


# prep data to send to javascript
class AttentionHeadImageJSInfo:

    def __init__(self, attn_head, image, name="No Name", cls_token=True):

        normalized_ah = normalize_attn_head(attn_head)
        if not cls_token:
            normalized_ah = normalized_ah[1:, 1:]

        image_size = image.shape[-1]
        assert image_size == image.shape[-2], "images are assumed to be square"

        patch_size = int(image_size // np.sqrt(len(normalized_ah) - 1))
        image = prepare_image(image)
        flattened_patches = flatten_into_patches(image, patch_size, image_size)

        self.patches = flattened_patches
        self.image_size = image_size
        self.attn_head = normalized_ah.tolist()
        self.name = name


def plot_attention_heads(attn_heads, images, names=None, ATTN_SCALING=8, cls_token=True):
    # Convert attn_heads and images to lists if they are not already
    if not isinstance(attn_heads, list):
        attn_heads = [attn_heads]
    if not isinstance(images, list):
        images = [images]

    # Create AttentionHeadImageJSInfo instances
    attn_head_image_js_infos = []
    for attn_head, image, name in zip(attn_heads, images, names or [None] * len(attn_heads)):
        info = AttentionHeadImageJSInfo(attn_head, image, name=name or "Attention Head", cls_token=cls_token)
        attn_head_image_js_infos.append(info)

    # Create a figure and axes using Matplotlib
    num_subplots = len(attn_head_image_js_infos)
    fig, axes = plt.subplots(nrows=num_subplots, ncols=2, figsize=(10, 5 * num_subplots))

    # Handle the case when there is only one subplot
    if num_subplots == 1:
        axes = [axes]

    for i, info in enumerate(attn_head_image_js_infos):
        # Display the image on the left subplot
        axes[i][0].imshow(info.patches)
        axes[i][0].set_title(f"Image Patches - {info.name}")
        axes[i][0].axis('off')

        # Display the attention head heatmap on the right subplot
        im = axes[i][1].imshow(info.attn_head, cmap='viridis', vmin=0, vmax=1)
        axes[i][1].set_title(f"Attention Head - {info.name}")
        axes[i][1].axis('off')

        # Add a colorbar for the attention head heatmap
        cbar = fig.colorbar(im, ax=axes[i][1])
        cbar.ax.set_ylabel('Attention Weight', rotation=270, labelpad=20)

    plt.tight_layout()

    # Create an IPyWidgets output widget to display the figure
    out = widgets.Output()
    with out:
        plt.show()

    return out


# get layer 0 activations from running the model on the sample image.
def _get_layer_0_activations(model, image):
    layer_0_activations = None

    def hook_fn(m,i,o):
        nonlocal layer_0_activations
        layer_0_activations = i[0][0].cpu().numpy()

    handle = model.blocks[0].attn.attn_scores.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        model(torch.from_numpy(np.expand_dims(image,axis=0)).cuda())
    handle.remove()

    return layer_0_activations