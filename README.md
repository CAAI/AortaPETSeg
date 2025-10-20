# Aorta PET Segmentation
![AortaPETSeg](https://github.com/CAAI/AortaPETSeg/blob/master/image.jpg?raw=true)

## Installation
To install this library
```cmd
python -m venv .aortapetseg_venv # (OPTIONAL)
source .aortapetseg_venv/bin/activate # (OPTIONAL)

pip install git+https://github.com/CAAI/AortaPETSeg.git
```

Set nnU-Net environment variables (in .bashrc) ([nnUnet Documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md))

```cmd
if [ -z ${nnUNet_raw} ]; then export nnUNet_raw="${nnUNet_raw_data_base}/nnUNet_raw"; fi
if [ -z ${nnUNet_preprocessed} ]; then export nnUNet_preprocessed="${nnUNet_raw_data_base}/nnUNet_preprocessed"; fi
if [ -z ${nnUNet_results} ]; then export nnUNet_results="${nnUNet_raw_data_base}/nnUNet_results"; fi
```

## Usage

```
python main.py [suv_pet_file_path] [out_seg_file_path]
```

## Preprocessing
Data must be converted to Standardized Uptake Value (SUV) of the first 40s of data.