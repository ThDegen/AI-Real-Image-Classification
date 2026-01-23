from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import typer
from PIL import Image


def _load_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _plot_image_grid(
    images: Sequence[Image.Image],
    titles: Sequence[str],
    output_path: Path,
    grid_size: int = 5,
) -> None:
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    for ax in axes[len(images) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def dataset_statistics(datadir: str = "data") -> None:
    """Compute dataset statistics and save figures."""
    data_dir = Path(datadir)
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    print("Train dataset")
    print(f"Number of images: {len(train_df)}")
    first_train_path = data_dir / train_df.iloc[0]["file_name"]
    first_train_image = _load_image(first_train_path)
    print(f"Image shape: {first_train_image.size[::-1]} (H, W)")
    print("\n")

    print("Test dataset")
    print(f"Number of images: {len(test_df)}")
    first_test_path = data_dir / test_df.iloc[0]["id"]
    first_test_image = _load_image(first_test_path)
    print(f"Image shape: {first_test_image.size[::-1]} (H, W)")
    print("\n")

    # Sample images from train set
    train_samples = train_df.head(25)
    train_images = [
        _load_image(data_dir / row["file_name"]) for _, row in train_samples.iterrows()
    ]
    train_titles = [f"Label: {row['label']}" for _, row in train_samples.iterrows()]
    _plot_image_grid(
        train_images,
        train_titles,
        Path("train_sample_images.png"),
    )

    # Sample images from test set
    test_samples = test_df.head(25)
    test_images = [
        _load_image(data_dir / row["id"]) for _, row in test_samples.iterrows()
    ]
    test_titles = ["Test"] * len(test_samples)
    _plot_image_grid(
        test_images,
        test_titles,
        Path("test_sample_images.png"),
    )

    # Train label distribution
    label_distribution = train_df["label"].value_counts().sort_index()
    plt.bar(label_distribution.index.astype(str), label_distribution.values)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("train_label_distribution.png")
    plt.close()


if __name__ == "__main__":
    typer.run(dataset_statistics)