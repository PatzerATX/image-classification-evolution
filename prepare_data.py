"""Download, split, inspect, and visualize MNIST and CIFAR-10."""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST


SEED = 4337
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")


def set_random_seed(seed):
    """Make random operations repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_training_data(dataset):
    """Split a dataset into 80% training and 20% validation data."""
    indices = np.arange(len(dataset))
    targets = np.asarray(dataset.targets)
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=SEED,
        stratify=targets,
    )
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def get_subset_targets(dataset_subset):
    """Get the class labels belonging to a PyTorch Subset."""
    all_targets = np.asarray(dataset_subset.dataset.targets)
    return all_targets[np.asarray(dataset_subset.indices)]


def save_sample_images(dataset, class_names, dataset_name):
    """Save five example training images from every class."""
    targets = get_subset_targets(dataset)
    figure, axes = plt.subplots(10, 5, figsize=(10, 16))

    for class_index, class_name in enumerate(class_names):
        sample_positions = np.flatnonzero(targets == class_index)[:5]
        for column, position in enumerate(sample_positions):
            image, _ = dataset[int(position)]
            image_to_show = image.permute(1, 2, 0).numpy().squeeze()
            axes[class_index, column].imshow(
                image_to_show,
                cmap="gray" if image.shape[0] == 1 else None,
            )
            axes[class_index, column].axis("off")
            if column == 0:
                axes[class_index, column].set_title(class_name, loc="left")

    figure.suptitle(f"{dataset_name} training samples")
    figure.tight_layout()
    file_name = dataset_name.lower().replace("-", "_") + "_training_samples.png"
    figure.savefig(OUTPUT_DIR / file_name, dpi=180)
    plt.close(figure)


def save_class_distribution(dataset, class_names, dataset_name):
    """Save a chart showing the number of training images in each class."""
    targets = get_subset_targets(dataset)
    class_counts = np.bincount(targets, minlength=len(class_names))

    figure, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(x=class_names, y=class_counts, ax=axis, color="#2f6f8f")
    axis.set(
        title=f"{dataset_name} training class distribution",
        xlabel="Class",
        ylabel="Images",
    )
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    file_name = dataset_name.lower().replace("-", "_") + "_class_distribution.png"
    figure.savefig(OUTPUT_DIR / file_name, dpi=180)
    plt.close(figure)


def summarize_dataset(name, train_data, validation_data, test_data, class_names):
    """Return the dataset facts required by the project description."""
    sample_image, _ = train_data[0]
    train_targets = get_subset_targets(train_data)
    return {
        "dataset": name,
        "image_shape": " x ".join(str(size) for size in sample_image.shape),
        "classes": len(class_names),
        "training_images": len(train_data),
        "validation_images": len(validation_data),
        "test_images": len(test_data),
        "smallest_training_class": int(np.bincount(train_targets).min()),
        "largest_training_class": int(np.bincount(train_targets).max()),
    }


def main():
    set_random_seed(SEED)
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Random seed: {SEED}")
    print(f"PyTorch device available: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("Downloading and loading MNIST and CIFAR-10...")

    to_tensor = transforms.ToTensor()

    mnist_full = MNIST(DATA_DIR, train=True, download=True, transform=to_tensor)
    mnist_test = MNIST(DATA_DIR, train=False, download=True, transform=to_tensor)
    cifar_full = CIFAR10(DATA_DIR, train=True, download=True, transform=to_tensor)
    cifar_test = CIFAR10(DATA_DIR, train=False, download=True, transform=to_tensor)

    mnist_train, mnist_validation = split_training_data(mnist_full)
    cifar_train, cifar_validation = split_training_data(cifar_full)

    datasets = [
        ("MNIST", mnist_train, mnist_validation, mnist_test, [str(i) for i in range(10)]),
        ("CIFAR-10", cifar_train, cifar_validation, cifar_test, cifar_full.classes),
    ]

    summaries = []
    for name, train_data, validation_data, test_data, class_names in datasets:
        summaries.append(
            summarize_dataset(
                name,
                train_data,
                validation_data,
                test_data,
                class_names,
            )
        )
        save_sample_images(train_data, class_names, name)
        save_class_distribution(train_data, class_names, name)

    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)

    print("\nDataset summary")
    print(summary_table.to_string(index=False))
    print(f"\nSaved preparation outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
