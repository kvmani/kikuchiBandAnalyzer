# Kikuchi Band Analyzer & CycleGAN Integration Guide

This guide provides a step‑by‑step walkthrough for converting Kikuchi band diffraction patterns into clean, reconstructed images using **kikuchiBandAnalyzer** and a CycleGAN model from **pytorch‑CycleGAN‑and‑pix2pix**.  The goal is to help new users export EBSD patterns, run them through a machine‑learning model, and reconstruct the processed data for further analysis.

## Prerequisites

Before starting, ensure the following prerequisites are met:

* **Python environment** – both projects rely on Python (≥3.7) and standard scientific libraries.  Activate the conda/virtual environment associated with your projects or install dependencies as described in each repository’s documentation.
* **EBSD data** – you should have `.ang` and `.oh5` files with the same base name inside one folder.  These files contain the electron backscatter diffraction patterns you want to process.
* **Repository clones** – clone the two repositories into local folders, for example:
  ```bash
  git clone https://github.com/kvmani/kikuchiBandAnalyzer.git
  git clone https://github.com/kvmani/pytorch-CycleGAN-and-pix2pix.git
  ```
* **Model weights** – the CycleGAN repository must contain the trained model weights under `checkpoints/cyclegan_kikuchi_model_weights/sim_kikuchi_no_preprocess_lr2e‑4_decay_400`.  If these files are not present, download them from your research storage.

## Workflow overview

The workflow consists of six main steps:

1. **Prepare the EBSD data folder** (contains your `.ang` and `.oh5` files).
2. **Export EBSD patterns to images** using `hdf5_image_export_and_validation.py` in the `kikuchiBandAnalyzer` project (export stage).
3. **Run CycleGAN inference** on the exported images via `run_kikuchi_inference.py` from the `pytorch‑CycleGAN‑and‑pix2pix` project.
4. **Reconstruct the processed patterns** by rerunning `hdf5_image_export_and_validation.py` in reconstruction mode with the processed images directory.
5. **Generate a modified `.oh5` file** and rename a copy of the original `.ang` file.
6. **Measure band widths** using `KikuchiBandWidthAutomator.py` with an updated options file.

Detailed instructions for each step follow.

## Step 1 – Organise EBSD data

Place your `.ang` and `.oh5` files in a dedicated folder.  Both files must share the same base name (e.g. `sample.ang` and `sample.oh5`).  This guide assumes you work on Windows; adjust file paths accordingly for Linux/Mac.

## Step 2 – Export EBSD patterns (export stage)

1. Open the `kikuchiBandAnalyzer` project in your IDE or a terminal.  The script to run is `hdf5_image_export_and_validation.py`.
2. Edit the configuration inside the script to point to your data.  The relevant section looks like:
   ```python
   config = {
       "hdf5_file_path": r"C:\\Users\\your_user\\Documents\\ml_data\\debarnaData\\strain_transpose_coarsened_coarsened.oh5",
       # dataset_path inside the .oh5 file, taken from EBSD/Data/Pattern
       "dataset_path": "strain_transpose_coarsened_coarsened/EBSD/Data/Pattern",
       "output_dir": "exported_images/30_12_24_magnetite_data_coarsened",
       "convert_to_8bit": True,
       "convert_back_to_16bit": True,
       "skip_image_export": False,
       "stage": "export",  # **export** stage exports EBSD patterns to PNGs
       # ... further options omitted for brevity ...
   }
   ```
   **Replace the bold paths** with the actual location of your `.oh5` file and choose a descriptive `output_dir` name.  When `stage` is set to `"export"` the script writes each diffraction pattern to a PNG file in the specified `output_dir` and prints a summary.
3. Run the script from within your Python environment:
   ```bash
   cd kikuchiBandAnalyzer
   python hdf5_image_export_and_validation.py
   ```
   The terminal output should report that images were converted and exported.  A new folder appears under `exported_images` with your chosen name.  To view this folder in your file explorer, right‑click it in the IDE and choose **“Open in Explorer”** (see the highlighted example below).

![Exported images folder highlighted in the IDE](assets/step3_export_folder.png)

## Step 3 – Run CycleGAN inference

With the exported EBSD PNGs ready, process them through the CycleGAN model:

1. Open the `pytorch‑CycleGAN‑and‑pix2pix` project in your IDE.  Locate the file `run_kikuchi_inference.py`.
2. Launch a terminal inside this project.  In the screenshot below, the red arrow points to the **Terminal** button in PyCharm/VS Code.  Clicking this icon opens a terminal panel at the bottom of the window.

![Opening the terminal in the IDE](assets/step4_open_terminal.png)

3. In the terminal, execute the following command as a single line (replace the paths with your own):
   ```bash
   python .\run_kikuchi_inference.py \
       --model_name cyclegan_kikuchi_model_weights/sim_kikuchi_no_preprocess_lr2e-4_decay_400 \
       --input_folder "C:\\Users\\your_user\\PycharmProjects\\kikuchiBandAnalyzer\\exported_images\\30_12_24_magnetite_data_coarsened" \
       --results_dir "debarna_magnetite_30_12_24_coarsened_A"
   ```
   **Do not insert line breaks** when running the command; the backslashes here illustrate continuation only.  The arguments mean:
   * `--model_name` – relative path to the CycleGAN model weights.
   * `--input_folder` – location of the exported PNG images from step 2.
   * `--results_dir` – name of the directory where inference results will be stored.

   The script loads the model, processes all input images, and writes results under `results_dir`.  Processing time depends on the number of images and your GPU/CPU.
4. Once finished, examine the results folder in your project tree.  The processed images are nested in:

   `results_dir/cyclegan_kikuchi_model_weights/sim_kikuchi_no_preprocess_lr2e-4_decay_400/test_latest/images`

   Each file corresponds to an enhanced EBSD pattern.  The screenshot below shows an example of this directory tree after running the command.  The `images` folder contains PNG files named `EBSP_000001.png`, `EBSP_000002.png`, etc.

![Processed images generated by CycleGAN](assets/step4_inference_results.png)

## Step 4 – Reconstruct the processed patterns (reconstruction stage)

Now instruct `kikuchiBandAnalyzer` to reconstruct the processed patterns and generate a new `.oh5` file:

1. Return to the `kikuchiBandAnalyzer` project and open `hdf5_image_export_and_validation.py` again.
2. In the `config` dictionary, update the following entries:
   * **`processed_image_dir`** – set this to the full path of the `images` directory created in step 3.  This variable is **only used in the `reconstruction` stage** and should be defined exactly as shown below:
     ```python
     "processed_image_dir": r"C:\\Users\\your_user\\PycharmProjects\\pytorch-CycleGAN-and-pix2pix\\debarna_magnetite_30_12_24_coarsened_A\\cyclegan_kikuchi_model_weights\\sim_kikuchi_no_preprocess_lr2e-4_decay_400\\test_latest\\images",
     ```
   * **`stage`** – change from `"export"` to `"reconstruct"`.  When set to `reconstruct`, the script reads the processed images, reinserts them into the HDF5 structure, and writes a new `.oh5` file.

   The screenshot highlights both updates: the red arrow points to `processed_image_dir`, and the white arrow shows the `stage` set to `reconstruct`.

![Updating `processed_image_dir` and setting stage to reconstruct](assets/step5_config_reconstruct.png)

3. Save the changes and run the script again:
   ```bash
   cd kikuchiBandAnalyzer
   python hdf5_image_export_and_validation.py
   ```
   A new `.oh5` file will be created in the same directory as your original `.oh5`.  Its name follows the pattern `{basename}_AI_modified.oh5`, indicating that the patterns were reconstructed using AI.

4. Copy your original `.ang` file and rename the copy to `{basename}_AI_modified.ang`.  This pairing ensures that the new `.oh5` and `.ang` files have matching base names for downstream analysis.

## Step 5 – Run the band‑width analyzer

The final stage is to measure band widths in the reconstructed data:

1. In `kikuchiBandAnalyzer`, open `bandDetectorOptionsMagnetite.yml` (or the relevant options file for your material).  Adjust parameters such as `desired_hkl` and thresholds to suit your experiment.
2. Execute the band‑width automator script:
   ```bash
   python KikuchiBandWidthAutomator.py
   ```
   When prompted, select your `{basename}_AI_modified.oh5` and the corresponding `{basename}_AI_modified.ang` files.  The script computes band widths for all requested reflections and writes a new `.ang` file named `{basename}_desiredHkl_bandWidth.ang`.  This file can be opened in standard EBSD analysis software for further interpretation.

## Tips for beginners

* **Working directory matters** – run scripts from within the root of the respective projects so that relative paths resolve correctly.
* **Use raw strings for Windows paths** – prefix strings with `r` (e.g. `r"C:\\path\\to\\file"`) to avoid escape‑character issues.
* **Check dataset paths inside `.oh5` files** – if you’re unsure of the `dataset_path`, open the `.oh5` using `h5py` or an HDF5 viewer and locate the path to `EBSD/Data/Pattern`.
* **GPU acceleration** – the CycleGAN inference will be much faster on a computer with a CUDA‑enabled GPU.  Without a GPU the process may take several minutes per hundred patterns.
* **Troubleshooting** – if a script fails, read the error message carefully.  Missing directories, incorrect paths, or insufficient write permissions are common causes.

## Conclusion

Following this guide you can export EBSD patterns, enhance them using a CycleGAN, reconstruct the processed patterns into a new HDF5 file, and perform band‑width analysis.  The included screenshots illustrate key steps such as locating exported images, launching a terminal, examining processed results, and updating configuration parameters.  As you gain familiarity, you can automate parts of this workflow or adapt it for other materials by modifying the options files.
