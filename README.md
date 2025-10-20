## Installation
To install this library
```cmd
python -m venv .aortapetseg_venv # (OPTIONAL)
source .aortapetseg_venv/bin/activate # (OPTIONAL)

pip install git+https://github.com/CAAI/rh-report.git
```



## Usage

```
python main.py [suv_pet_file_path] [out_seg_file_path]
```

## Preprocessing
Data must be converted to Standardized Uptake Value (SUV) of the first 40s of data.