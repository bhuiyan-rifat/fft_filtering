# Z-scan FFT Filtering and Fitting Pipeline

This repository provides a Python script for **FFT-based denoising of Z-scan data**.  
It is designed to work in **batch mode** over multiple materials and power values.

## Command Usage

    python main.py --base_path /path/to/data --materials AristoD3 Calcirol

### Arguments

- `--base_path /path/to/data`  
  Path to the parent directory that contains all material folders.

- `--materials AristoD3 Calcirol`  
  One or more material names (space-separated).  
  Each name must match a folder inside `base_path`.

---

### Example

If your data is stored as:

```
C:/data/zscan/
├── AristoD3/
├── Calcirol/
└── D-Rise/
```

To process **AristoD3** and **Calcirol**, run:

```
python main.py --base_path C:/data/zscan --materials AristoD3 Calcirol
```

The script will process both materials and save results into:

```
C:/data/zscan/fft_data/
```


## 1. Input Data Structure (IMPORTANT)

The code assumes a **specific directory structure** for batch processing.

### Base Path

The user provides a single **base path** that contains **multiple material folders**.

```
base_path/
├── AristoD3/
├── Calcirol/
├── D-Rise/
├── Defrol/
└── OsteoD/
```

---

### Material Folders

Each **material folder** must contain CSV files named using the pattern:

```
<POWER>_ca_oa.csv
```

Where:
- `<POWER>` is an integer power value
- Expected power range: **185 to 254 (inclusive)**

Example:

```
AristoD3/
├── 185_ca_oa.csv
├── 186_ca_oa.csv
├── 187_ca_oa.csv
│   ...
└── 254_ca_oa.csv
```

⚠️ **File naming must match exactly**, otherwise the script may fail or skip files.

---

## 2. Output Structure

When the script is run, it **automatically creates an output directory** inside the base path.

```
base_path/
└── fft_data/
    ├── AristoD3/
    ├── Calcirol/
    ├── D-Rise/
    └── ...
```

---

### Output Files per Power Value

For **each `<POWER>_ca_oa.csv`**, the following files are generated:

| File Name Suffix              | Description |
|-------------------------------|-------------|
| `_original_normalized.csv`    | Normalized original signal |
| `_filtered.csv`               | FFT-denoised signal |
| `_fitted.csv`                 | Model-fitted signal |
| `_meta.csv`                   | Metadata and fitting parameters |
| `_plot.pdf`                   | Plot or original and filtered data |
