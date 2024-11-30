import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random

from pathlib import Path
import h5py
import numpy as np
from PIL import Image


def export_images_from_hdf5(hdf5_file_path, output_dir, dataset_path, base_name="pixel", index_format="%05d",
                            file_extension=".png"):
    """
    Export images from an HDF5 dataset to individual image files.

    Parameters:
        hdf5_file_path (str): Path to the HDF5 file.
        output_dir (str): Directory to save the exported images.
        dataset_path (str): Path to the dataset inside the HDF5 file.
        base_name (str): Base name for the output image files.
        index_format (str): Format for zero-padded integer indices, e.g., "%05d".
        file_extension (str): File extension for the output image files.

    Returns:
        data (numpy.ndarray): The dataset array.
        output_dir (Path): Path object of the output directory.
    """
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
            formatted_index = index_format % (i+1)
            image_filename = f"{base_name}_{formatted_index}{file_extension}"
            image_path = output_dir / image_filename
            # Ensure the image is saved as 16-bit if necessary
            Image.fromarray(image.astype(np.uint16)).save(image_path)

        print(f"Exported {N} images to {output_dir}")

        return data, output_dir


def load_images_to_array(output_dir, N, m, n, base_name="pixel", index_format="%05d", file_extension=".png"):
    """
    Load images from a directory into a numpy array.

    Parameters:
        output_dir (str or Path): Directory where the images are stored.
        N (int): Number of images to load.
        m (int): Height of each image.
        n (int): Width of each image.
        base_name (str): Base name of the images.
        index_format (str): Format for zero-padded integer indices, e.g., "%05d".
        file_extension (str): File extension of the image files.

    Returns:
        loaded_array (numpy.ndarray): Loaded images as a numpy array.
    """
    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)

    # Initialize an array to store loaded images
    loaded_array = np.zeros((N, m, n), dtype=np.uint16)

    # Load each image back
    for i in range(N):
        formatted_index = index_format % (i+1)
        image_filename = f"{base_name}_{formatted_index}{file_extension}"
        image_path = output_dir / image_filename
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
output_dir = Path("exported_images/big_grain")
base_name="big_grain"

# Run the process
original_array, output_dir = export_images_from_hdf5(
    hdf5_file_path, output_dir, dataset_path, base_name=base_name
)
N, m, n = original_array.shape
reconstructed_array = load_images_to_array(output_dir, N, m, n,
                                           base_name=base_name)

# Check if the arrays are equal
arrays_equal = compare_arrays(original_array, reconstructed_array)
print(f"Arrays match: {arrays_equal}")

# Plot random images to double-check
plot_random_images(original_array, reconstructed_array)
