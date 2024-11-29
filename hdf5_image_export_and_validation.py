import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random


def export_images_from_hdf5(hdf5_file_path, output_dir, dataset_path, basename_format="pixel_{index}.png"):
    # Open the HDF5 file
    with h5py.File(hdf5_file_path, 'r') as hdf:
        # Access the dataset
        data = hdf[dataset_path][()]
        N, m, n = data.shape

        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export each image
        for i in range(N):
            image = data[i]
            image_path = output_dir / basename_format.format(index=i)
            # Ensure the image is saved as 16-bit if necessary
            Image.fromarray(image.astype(np.uint16)).save(image_path)
        print(f"Exported {N} images to {output_dir}")

        return data, output_dir


def load_images_to_array(output_dir, N, m, n, basename_format="pixel_{index}.png"):
    # Initialize an array to store loaded images
    loaded_array = np.zeros((N, m, n), dtype=np.uint16)

    # Load each image back
    for i in range(N):
        image_path = output_dir / basename_format.format(index=i)
        loaded_array[i] = np.array(Image.open(image_path), dtype=np.uint16)

    return loaded_array


def compare_arrays(original_array, reconstructed_array):
    # Compare the arrays and return the result
    return np.array_equal(original_array, reconstructed_array)


def plot_random_images(original_array, reconstructed_array, num_images=5):
    N = original_array.shape[0]
    indices = random.sample(range(N), min(num_images, N))

    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))
    for i, idx in enumerate(indices):
        axes[i, 0].imshow(original_array[idx], cmap='gray')
        axes[i, 0].set_title(f"Original Image {idx}")
        axes[i, 1].imshow(reconstructed_array[idx], cmap='gray')
        axes[i, 1].set_title(f"Reconstructed Image {idx}")

    plt.tight_layout()
    plt.show()


# File paths and dataset details
hdf5_file_path = "C:\\Users\\kvman\\Downloads\\New_regions\\New_regions\\big_grain.oh5"
dataset_path = "big_grain/EBSD/Data/Pattern"
output_dir = Path("exported_images")

# Run the process
original_array, output_dir = export_images_from_hdf5(
    hdf5_file_path, output_dir, dataset_path
)
N, m, n = original_array.shape
reconstructed_array = load_images_to_array(output_dir, N, m, n)

# Check if the arrays are equal
arrays_equal = compare_arrays(original_array, reconstructed_array)
print(f"Arrays match: {arrays_equal}")

# Plot random images to double-check
plot_random_images(original_array, reconstructed_array)
