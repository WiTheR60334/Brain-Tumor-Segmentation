# Brain Tumor Segmentation Model with UNet

This repository contains a brain tumor segmentation model built using UNet. The model is trained to segment brain tumor regions from MRI images using a dataset that includes various MRI modalities.

## Dataset

The brain tumor dataset used in this project can be downloaded from [Synapse](https://www.synapse.org/Synapse:syn53708126/wiki/626320). The dataset contains the following files for each tumor image:

- **Segmentation File (Ground Truth)**
- **T2F (T2-Flair)**
- **T1C (T1-Weighted with Contrast)**
- **T1N (T1-Weighted without Contrast)**
- **T2W (T2-Weighted)**

## Getting Started

### Step 1: Download the Dataset

1. Download the brain tumor dataset from [Synapse](https://www.synapse.org/Synapse:syn53708126/wiki/626320).
2. Ensure the dataset includes the segmentation file and the MRI modalities mentioned above.

### Step 2: Prepare the Data

1. Change all file paths in the code to match your local file paths.
2. Run `data_init.ipynb`:
   - This notebook will visually plot brain tumor images and save the mask file along with a combined file (Flair, Contrast, and Weighted) of the brain tumor.
   - The notebook contains two parts:
     1. First part for trial with one image.
     2. Second part saves all the masks of all brain tumor files in a folder.
3. After executing `data_init.ipynb`, two folders will be created:
   - `NPYcombined`
   - `NPYmasked`

### Step 3: View the Combined and Masked NPYs

- Run `NPYviewer.ipynb` to view the combined and masked NPY files.

### Step 4: Train the Model

- Run `main.ipynb` to train the UNet model and check the predictions.

### Additional Step: Downloading the Dataset Using `downloader.py`

- Use `downloader.py` to download the BRATS dataset from Synapse.org.
- Steps:
  1. Log in to Synapse.org.
  2. Copy your authToken.
  3. Paste the authToken into `downloader.py`.
  4. Run `downloader.py` to download the dataset.

## Repository Structure

- `data_init.ipynb`: Prepares and visualizes the data.
- `NPYviewer.ipynb`: Views the combined and masked NPY files.
- `main.ipynb`: Trains the UNet model and makes predictions.
- `downloader.py`: Script to download the dataset from Synapse.org.

Install the dependencies using:

```bash
pip install -r requirements.txt
