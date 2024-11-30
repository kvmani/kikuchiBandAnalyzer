import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random
import logging
import shutil
import os


# Configure logging
logging.basicConfig(level=logging.INFO)


def export_images_from_hdf5(hdf5_file_path, output_dir, dataset_path, options):
    base_name = options.get("base_name", "pixel")
    index_format = options.get("index_format", "%05d")
    file_extension = options.get("file_extension", ".png")
    convert_to_8bit = options.get("convert_to_8bit", True)

    with h5py.File(hdf5_file_path, 'r') as hdf:
        data = hdf[dataset_path][()]
        N, m, n = data.shape

        output_dir = Path(output_dir)
        if convert_to_8bit:
            logging.warning("Converting 16-bit image to 8-bit for export.")

        if not options["skip_image_export"]:
            output_dir.mkdir(parents=True, exist_ok=True)

            for i in range(N):
                image = data[i]
                if convert_to_8bit:
                    max_val = image.max()
                    image = ((image / max_val) * 255).astype(np.uint8)


                formatted_index = index_format % (i + 1)
                image_filename = f"{base_name}_{formatted_index}{file_extension}"
                image_path = output_dir / image_filename
                Image.fromarray(image).save(image_path)
        else:
            logging.info("Skipping the image export part!!!!")

        logging.info(f"Exported {N} images to {output_dir}")
        return data, output_dir


def load_images_to_array(output_dir, N, m, n, options):
    base_name = options.get("base_name", "pixel")
    index_format = options.get("index_format", "%05d")
    file_extension = options.get("file_extension", ".png")
    convert_to_16bit = options.get("convert_to_16bit", True)

    output_dir = Path(output_dir)
    loaded_array = np.zeros((N, m, n), dtype=np.uint16)
    if convert_to_16bit:
        #image = (image / 255 * 65535).astype(np.uint16)
        logging.warning("Converting 8-bit image back to 16-bit.")

    for i in range(N):
        formatted_index = index_format % (i + 1)
        image_filename = f"{base_name}_{formatted_index}{file_extension}"
        image_path = output_dir / image_filename
        image = np.array(Image.open(image_path), dtype=np.uint16)

        if convert_to_16bit:
            image = (image / 255 * 65535).astype(np.uint16)
            #logging.warning("Converting 8-bit image back to 16-bit.")

        loaded_array[i] = image

    return loaded_array


def make_modified_hdf(original_hdf_path, new_array, dataset_path, output_path):
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


def repack_hdf5_inplace(hdf_file_path):
    """
    Repack an HDF5 file in-place to remove unused space and optimize storage.

    Parameters:
        hdf_file_path (str): Path to the HDF5 file to be repacked.

    Returns:
        None
    """
    tmp_file = Path(hdf_file_path).with_suffix(".tmp.oh5")

    # Create the temporary repacked file
    with h5py.File(hdf_file_path, "r") as src, h5py.File(tmp_file, "w") as dst:
        def copy_attrs(source, target):
            for key, value in source.attrs.items():
                target.attrs[key] = value

        def copy_data(src_group, dst_group):
            for name, item in src_group.items():
                if isinstance(item, h5py.Group):
                    new_group = dst_group.create_group(name)
                    copy_attrs(item, new_group)
                    copy_data(item, new_group)
                elif isinstance(item, h5py.Dataset):
                    data = item[()]
                    compression = item.compression
                    chunks = item.chunks
                    new_dataset = dst_group.create_dataset(
                        name, data=data, compression=compression, chunks=chunks
                    )
                    copy_attrs(item, new_dataset)

        copy_data(src, dst)

    # Ensure the target file is removed before renaming the temp file
    hdf_file_path = Path(hdf_file_path)
    if hdf_file_path.exists():
        hdf_file_path.unlink()

    # Rename the temporary file to the original file
    tmp_file.rename(hdf_file_path)
    logging.info(f"Repacked file saved in place at {hdf_file_path}")


def compare_hdf_sizes(original_file, modified_file):
    original_size = os.path.getsize(original_file)
    modified_size = os.path.getsize(modified_file)

    logging.info(f"Original file size: {original_size / 1e6:.2f} MB")
    logging.info(f"Modified file size: {modified_size / 1e6:.2f} MB")
    size_diff = modified_size - original_size
    logging.info(f"Size difference: {size_diff / 1e6:.2f} MB")


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
    "skip_image_export": True,
}
#### case of magnetite_data_coarsened (small test data)
hdf5_file_path = "C:\\Users\\kvman\\Downloads\\magnetite_data_coarsened.oh5"
dataset_path = "magnetite data coarsened coarsened/EBSD/Data/Pattern"
output_dir = "exported_images/magnetite_data_coarsened"
input_image_folder = (r"C:\Users\kvman\PycharmProjects\pytorch-CycleGAN-and-pix2pix"
                      r"\debarna_test\cyclegan_kikuchi_model_weights"
                      r"\sim_kikuchi_no_preprocess_lr2e-4_decay_400\test_latest\images")

#case of large magnetite_data
# Example Configurations
options = {
    "base_name": "magnetite_data",
    "index_format": "%05d",
    "file_extension": ".png",
    "convert_to_8bit": True,
    "convert_to_16bit": True,
    "skip_image_export": True,
}
hdf5_file_path = "C:/Users/kvman/Downloads/OneDrive_1_10-20-2024/magnetite_data.oh5"
dataset_path = "magnetite data/EBSD/Data/Pattern"
output_dir = "exported_images/magnetite_data"
input_image_folder = (r"C:\Users\kvman\PycharmProjects\pytorch-CycleGAN-and-pix2pix"
                      r"\debarna_magnetite_ai_processed\cyclegan_kikuchi_model_weights"
                      r"\sim_kikuchi_no_preprocess_lr2e-4_decay_400\test_latest\images")

original_array, output_dir = export_images_from_hdf5(hdf5_file_path, output_dir, dataset_path, options)
if options["skip_image_export"]:
    output_dir = input_image_folder
    N, m, n = original_array.shape
    reconstructed_array = load_images_to_array(output_dir, N, m, n, options)
    input_hdf_dir = Path(hdf5_file_path).parent
    modified_hdf_path = input_hdf_dir / f"{Path(hdf5_file_path).stem}_AI_modified.oh5"

    modified_hdf_path = make_modified_hdf(
        hdf5_file_path,
        reconstructed_array,
        dataset_path,
        modified_hdf_path
    )

    # Repack the modified HDF5 file in place
    repack_hdf5_inplace(modified_hdf_path)

    # Compare sizes
    compare_hdf_sizes(hdf5_file_path, modified_hdf_path)

    # Verify and plot
    arrays_equal = compare_arrays(original_array, reconstructed_array)
    print(f"Arrays match: {arrays_equal}")
    plot_random_images(original_array, reconstructed_array)
