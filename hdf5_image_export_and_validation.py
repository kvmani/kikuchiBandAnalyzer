import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random
import logging
import shutil


# Configure logging
logging.basicConfig(level=logging.INFO)


def export_images_from_hdf5(hdf5_file_path, output_dir, dataset_path, options):
    """
    Export images from an HDF5 dataset to individual image files, optionally converting to 8-bit.

    Parameters:
        hdf5_file_path (str): Path to the HDF5 file.
        output_dir (str): Directory to save the exported images.
        dataset_path (str): Path to the dataset inside the HDF5 file.
        options (dict): Dictionary with options such as base_name, index_format, file_extension, convert_to_8bit.

    Returns:
        data (numpy.ndarray): The dataset array.
        output_dir (Path): Path object of the output directory.
    """
    base_name = options.get("base_name", "pixel")
    index_format = options.get("index_format", "%05d")
    file_extension = options.get("file_extension", ".png")
    convert_to_8bit = options.get("convert_to_8bit", True)

    with h5py.File(hdf5_file_path, 'r') as hdf:
        data = hdf[dataset_path][()]
        N, m, n = data.shape

        output_dir = Path(output_dir)
        if options["skip_image_export"]:
            output_dir.mkdir(parents=True, exist_ok=True)

            for i in range(N):
                image = data[i]
                if convert_to_8bit:
                    max_val = image.max()
                    image = ((image / max_val) * 255).astype(np.uint8)
                    logging.warning("Converting 16-bit image to 8-bit for export.")

                formatted_index = index_format % (i + 1)
                image_filename = f"{base_name}_{formatted_index}{file_extension}"
                image_path = output_dir / image_filename
                Image.fromarray(image).save(image_path)
        else:
            logging.info(f"Skipping the image export part!!!!")

        logging.info(f"Exported {N} images to {output_dir}")
        return data, output_dir


def load_images_to_array(output_dir, N, m, n, options):
    """
    Load images from a directory into a numpy array, optionally converting to 16-bit.

    Parameters:
        output_dir (str or Path): Directory where the images are stored.
        N (int): Number of images to load.
        m (int): Height of each image.
        n (int): Width of each image.
        options (dict): Dictionary with options such as base_name, index_format, file_extension, convert_to_16bit.

    Returns:
        loaded_array (numpy.ndarray): Loaded images as a numpy array.
    """
    base_name = options.get("base_name", "pixel")
    index_format = options.get("index_format", "%05d")
    file_extension = options.get("file_extension", ".png")
    convert_to_16bit = options.get("convert_to_16bit", True)

    output_dir = Path(output_dir)
    loaded_array = np.zeros((N, m, n), dtype=np.uint16)

    for i in range(N):
        formatted_index = index_format % (i + 1)
        image_filename = f"{base_name}_{formatted_index}{file_extension}"
        image_path = output_dir / image_filename
        image = np.array(Image.open(image_path), dtype=np.uint16)

        if convert_to_16bit:
            image = (image / 255 * 65535).astype(np.uint16)
            logging.warning("Converting 8-bit image back to 16-bit.")

        loaded_array[i] = image

    return loaded_array


def make_modified_hdf(original_hdf_path, new_array, dataset_path, output_path):
    """
    Generate a modified HDF5 file by replacing a specific dataset.

    Parameters:
        original_hdf_path (str): Path to the original HDF5 file.
        new_array (numpy.ndarray): Array to replace the dataset with.
        dataset_path (str): Path to the dataset inside the HDF5 file.
        output_path (str): Path for the new modified HDF5 file.

    Returns:
        str: Path to the modified HDF5 file.
    """
    output_path = Path(output_path)
    if output_path.exists():
        logging.warning(f"Output file {output_path} already exists and will be overwritten.")

    shutil.copy(original_hdf_path, output_path)
    logging.info(f"Copied original HDF5 file to {output_path}.")

    with h5py.File(output_path, "a", libver="latest") as hdf:
        if dataset_path in hdf:
            del hdf[dataset_path]
            logging.info(f"Deleted existing dataset {dataset_path}.")
        hdf.create_dataset(dataset_path, data=new_array)
        logging.info(f"Created new dataset at {dataset_path} with the modified array.")

    return str(output_path)


def compare_arrays(original_array, reconstructed_array):
    return np.array_equal(original_array, reconstructed_array)


def plot_random_images(original_array, reconstructed_array, num_images=2):
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


# Example Configurations
options = {
    "base_name": "magnetite_data_coarsened",
    "index_format": "%05d",
    "file_extension": ".png",
    "convert_to_8bit": True,
    "convert_to_16bit": True,
    "skip_image_export":True, ### set true if you already have images (say from AI output)
}

hdf5_file_path = "C:\\Users\\kvman\\Downloads\\magnetite_data_coarsened.oh5"
dataset_path = "magnetite data coarsened coarsened/EBSD/Data/Pattern"
output_dir = "exported_images/magnetite_data_coarsened"
input_image_folder=(r"C:\Users\kvman\PycharmProjects\pytorch-CycleGAN-and-pix2pix"
                    r"\debarna_test\cyclegan_kikuchi_model_weights"
                    r"\sim_kikuchi_no_preprocess_lr2e-4_decay_400\test_latest\images")

# Export images

original_array, output_dir = export_images_from_hdf5(hdf5_file_path, output_dir, dataset_path, options)
if options["skip_image_export"]:
    output_dir = input_image_folder
# Load images back
N, m, n = original_array.shape
reconstructed_array = load_images_to_array(output_dir, N, m, n, options)
input_hdf_dir = Path(hdf5_file_path).parent
modified_hdf_path = input_hdf_dir / f"{Path(hdf5_file_path).stem}_AI_modified.oh5"

# Create a modified HDF5 file
modified_hdf_path = make_modified_hdf(
    hdf5_file_path,
    reconstructed_array,
    dataset_path,
    modified_hdf_path
)

# Verify and plot
arrays_equal = compare_arrays(original_array, reconstructed_array)
print(f"Arrays match: {arrays_equal}")
plot_random_images(original_array, reconstructed_array)
