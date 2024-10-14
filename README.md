# Kikuchi Band Width Detector

This project aims to detect the width of Kikuchi bands in Electron Backscatter Diffraction (EBSD) images using various detection strategies. It processes multiple bands in a single image and can handle multiple images, using input parameters from a JSON file. The output includes useful information such as band width, central peak, and detection success for each band, saved in a JSON file.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Strategies](#strategies)
- [Output](#output)
- [Logging](#logging)
- [Example](#example)
- [Future Enhancements](#future-enhancements)

---

## Project Structure


---

## Installation

### Prerequisites

- Python 3.x
- Required packages:
    - `opencv-python`
    - `numpy`
    - `scipy`
    - `matplotlib`
    - `pyyaml`

To install the necessary packages, run:

```bash
pip install opencv-python numpy scipy matplotlib pyyaml


[
    {
        "grainId": 10,
        "grain_xy": [12, 45],
        "patternFileName": "testData/ML_kikuchi_test_1.png",
        "LRS_value": 1.5e6,
        "points": [
            {"hkl": "110", "central_line": [[65, 44], [237, 151]], "refWidth": 100},
            {"hkl": "220", "central_line": [[81, 63], [90, 48]], "refWidth": 120}
        ],
        "comment": "Big grain"
    },
    {
        "grainId": 11,
        "grain_xy": [56, 95],
        "patternFileName": "testData/poorKikuci.png",
        "LRS_value": -5.5e6,
        "points": [
            {"hkl": "111", "central_line": [[41, 100], [61, 100]], "refWidth": 105},
            {"hkl": "420", "central_line": [[107, 171], [102, 184]], "refWidth": 105}
        ],
        "comment": "Another grain"
    }
]

python kikuchiBandWidthDetector.py


gradient_threshold: 50
perpendicular_line_length: 40
smoothing_sigma: 2
strategy: "rectangular_area"   # Options: "gradient", "gaussian", "rectangular_area"
debug: true                    # Enable or disable debug mode
rectWidth: 50                  # Width of the rectangle for the rectangular_area strategy

python kikuchiBandWidthDetector.py


You can now copy and paste this content into your `README.md`. Let me know if you need further assistance!
