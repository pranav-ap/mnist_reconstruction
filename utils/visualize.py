import torch

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
import seaborn as sns
sns.set_theme(style="darkgrid")


def visualize_X_samples_grid(dataset, labels, n_samples=12, n_cols=4):
    n_rows = n_samples // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    for i, ax in enumerate(axes.flat):
        label = labels[i]
        img = dataset[i]

        if isinstance(img, torch.Tensor):  # Check if it's a tensor
            img = img.detach().numpy()  # Detach from graph if it requires grad

        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f"Label: {label}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


