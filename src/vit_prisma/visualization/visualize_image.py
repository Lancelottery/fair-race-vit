import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt

def display_grid_on_image(image, patch_size=32, return_plot=False):
    if isinstance(image, torch.Tensor):
        image = image.detach().numpy().transpose(1, 2, 0)
    if image.shape[0] != 224:
        image = image.transpose(1, 2, 0)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    num_patches = (image.shape[0] / patch_size) ** 2
    grid_size = int(np.sqrt(num_patches))

    # Calculate patch size
    patch_height = image.shape[0] // grid_size
    patch_width = image.shape[1] // grid_size

    # Overlay grid
    grid_image = np.copy(image)
    for i in range(1, grid_size):
        # Vertical lines
        grid_image[:, patch_width * i, :] = [255, 255, 255]
        # Horizontal lines
        grid_image[patch_height * i, :, :] = [255, 255, 255]

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)  # Adjust figsize and dpi as needed

    # Place labels
    for i in range(grid_size):
        for j in range(grid_size):
            x_center = (j * patch_width) + (patch_width // 2)
            y_center = (i * patch_height) + (patch_height // 2)
            # Convert the patch index to 2D coordinates (row, column)
            patch_index = i * grid_size + j
            row, col = divmod(patch_index, grid_size)
            ax.text(x_center, y_center, f"{patch_index+1}", color='red', fontsize=8, ha='center', va='center')

    # Display image with grid and labels
    ax.imshow(grid_image)
    ax.axis('off')

    if return_plot:
        return fig
    else:
        plt.show()


def display_grid_on_images(images, num_images, patch_size=32, return_plot=False):
    fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 4), dpi=100)
    
    if num_images == 1:  # Handle single image case
        axes = [axes]

    for idx, (image, ax) in enumerate(zip(images[:num_images], axes)):
        # Reverse the normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean  # Apply reverse normalization
        image = image.permute(1, 2, 0)  # Reorder dimensions for displaying
        image = image.clamp(0, 1)  # Clamp the values to [0, 1] to avoid warnings
        
        if isinstance(image, torch.Tensor):
            image = image.detach().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        num_patches = (image.shape[0] / patch_size) ** 2
        grid_size = int(np.sqrt(num_patches))

        # Calculate patch size
        patch_height = image.shape[0] // grid_size
        patch_width = image.shape[1] // grid_size

        # Overlay grid
        grid_image = np.copy(image)
        for i in range(1, grid_size):
            # Vertical lines
            grid_image[:, patch_width * i, :] = [255, 255, 255]
            # Horizontal lines
            grid_image[patch_height * i, :, :] = [255, 255, 255]

        # Place labels
        for i in range(grid_size):
            for j in range(grid_size):
                x_center = (j * patch_width) + (patch_width // 2)
                y_center = (i * patch_height) + (patch_height // 2)
                # Convert the patch index to 2D coordinates (row, column)
                patch_index = i * grid_size + j
                row, col = divmod(patch_index, grid_size)
                ax.text(x_center, y_center, f"{patch_index+1}", color='red', fontsize=8, ha='center', va='center')

        # Display image with grid and labels
        ax.imshow(grid_image)
        ax.axis('off')

    plt.tight_layout()
    if return_plot:
        return fig
    else:
        plt.show()
