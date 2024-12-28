import h5py
import numpy as np
import logging
from pathlib import Path
from PIL import Image
import random
import matplotlib.pyplot as plt
import shutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO)


class EBSDProcessor:
    def __init__(self, config):
        self.config = config

    def export_images(self):
        options = self.config["options"]
        dataset_path = self.config["dataset_path"]
        output_dir = Path(self.config["output_dir"])
        hdf5_file_path = self.config["hdf5_file_path"]

        with h5py.File(hdf5_file_path, 'r') as hdf:
            data = hdf[dataset_path][()]
            N, m, n = data.shape

            if options["convert_to_8bit"]:
                logging.info("Converting 16-bit images to 8-bit for export.")

            output_dir.mkdir(parents=True, exist_ok=True)
            for i in range(N):
                image = data[i]
                if options["convert_to_8bit"]:
                    image = ((image / image.max()) * 255).astype(np.uint8)

                file_name = f"{options['base_name']}_{options['index_format'] % (i + 1)}{options['file_extension']}"
                Image.fromarray(image).save(output_dir / file_name)
            logging.info(f"Exported {N} images to {output_dir}.")

        return data, output_dir

    def load_images(self):
        options = self.config["options"]
        output_dir = Path(self.config["processed_image_dir"])
        N, m, n = self.config["image_shape"]

        images = np.zeros((N, m, n), dtype=np.uint16)
        for i in range(N):
            file_name = f"{options['base_name']}_{options['index_format'] % (i + 1)}{options['file_extension']}"
            image = np.array(Image.open(output_dir / file_name), dtype=np.uint16)
            if options["convert_back_to_16bit"]:
                image = (image / 255 * 65535).astype(np.uint16)
            images[i] = image

        return images

    def modify_hdf5(self, new_array):
        hdf5_file_path = Path(self.config["hdf5_file_path"])
        output_path = hdf5_file_path.with_name(f"{hdf5_file_path.stem}_AI_modified.h5")
        dataset_path = self.config["dataset_path"]

        if output_path.exists():
            logging.warning(f"Overwriting existing file: {output_path}")

        shutil.copy(hdf5_file_path, output_path)
        with h5py.File(output_path, "a") as hdf:
            if dataset_path in hdf:
                del hdf[dataset_path]
            hdf.create_dataset(dataset_path, data=new_array)
        logging.info(f"Updated HDF5 file saved at {output_path}.")
        return output_path

    def repack_hdf5(self, path):
        tmp_file = Path(path).with_suffix(".tmp.h5")
        with h5py.File(path, "r") as src, h5py.File(tmp_file, "w") as dst:
            for key in src:
                src.copy(key, dst)

        Path(path).unlink()
        tmp_file.rename(path)
        logging.info(f"Repacked HDF5 file at {path}.")

    def compare_and_plot(self, original_array, reconstructed_array):
        arrays_match = np.array_equal(original_array, reconstructed_array)
        logging.info(f"Array comparison result: {arrays_match}")

        indices = random.sample(range(original_array.shape[0]), 2)
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for i, idx in enumerate(indices):
            axes[i][0].imshow(original_array[idx], cmap='gray')
            axes[i][0].set_title(f"Original Image {idx}")
            axes[i][1].imshow(reconstructed_array[idx], cmap='gray')
            axes[i][1].set_title(f"Reconstructed Image {idx}")
        plt.show()

    def run(self):
        if self.config["stage"] == "export":
            self.export_images()
        elif self.config["stage"] == "reconstruct":
            #original_array, _ = self.export_images()
            reconstructed_array = self.load_images()
            #self.modify_hdf5(reconstructed_array)
            modified_hdf_path = self.modify_hdf5(reconstructed_array)
            # Repack the modified HDF5 file
            self.repack_hdf5(modified_hdf_path)

            #self.compare_and_plot(original_array, reconstructed_array)


if __name__ == "__main__":
    # Configuration dictionary with your provided values
    config = {
        "hdf5_file_path": "C:\\Users\\kvman\\Downloads\\magnetite_data_coarsened.oh5",
        "dataset_path": "magnetite data coarsened coarsened/EBSD/Data/Pattern",
        "output_dir": "exported_images/magnetite_data_coarsened",
        "processed_image_dir": (r"C:\Users\kvman\PycharmProjects\pytorch-CycleGAN-and-pix2pix"
                                r"\debarna_test\cyclegan_kikuchi_model_weights"
                                r"\sim_kikuchi_no_preprocess_lr2e-4_decay_400\test_latest\images"),
        "image_shape": (100, 200, 200),  # Example shape (adjust based on data)
        #"stage": "export",  # "export" or "reconstruct"
        "stage": "reconstruct",  # "export" or "reconstruct"
        "options": {
            "base_name": "magnetite_data_coarsened",
            "index_format": "%05d",
            "file_extension": ".png",
            "convert_to_8bit": True,
            "convert_back_to_16bit": True,
            "skip_image_export": True,
        }
    }

    config = {
        "hdf5_file_path": r"C:\Users\kvman\Documents\ml_data\debarnaData\strain_transpose_coarsened_coarsened.oh5",
        "dataset_path": "strain_transpose coarsened coarsened/EBSD/Data/Pattern",
        "output_dir": "exported_images/28_12_24_magnetite_data_coarsened",
        "processed_image_dir": r"C:\Users\kvman\PycharmProjects\pytorch-CycleGAN-and-pix2pix"
                               r"\debarna_magnetite_28_12_24_coarsened_AI\cyclegan_kikuchi_model_weights"
                               r"\sim_kikuchi_no_preprocess_lr2e-4_decay_400\test_latest\images",
        "image_shape": (12463, 230, 230),  # Example shape (adjust based on data)
        "stage": "reconstruct",  # "export" or "reconstruct"
        "options": {
            "base_name": "28_12_24_magnetite_coarsened",
            "index_format": "%05d",
            "file_extension": ".png",
            "convert_to_8bit": True,
            "convert_back_to_16bit": True,
            "skip_image_export": True,
        }
    }


    processor = EBSDProcessor(config)
    processor.run()
