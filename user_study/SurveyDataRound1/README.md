# Round 1 User Study – Analysis Artifact

This folder contains the analysis code and data for **Round 1** of our user study.

All analyses are implemented in Jupyter notebooks and can be reproduced by running the notebooks from top to bottom.

## 1. Folder structure

```
SurveyDataRound1/
│
├── ComputerRound1.ipynb                 # Analysis for computer condition
├── PhoneRound1.ipynb                    # Analysis for smartphone condition
│
├── Adversarial-round1-cleaned.csv       # Cleaned response data (computer)
├── Adversarial-round1-phone-cleaned.csv # Cleaned response data (phone)
├── all-computer-statistic.csv           # Pre-computed statistics (computer)
├── all-phone-statistic.csv              # Pre-computed statistics (phone)
├── computer_rule_match.csv              # Mapping between items and rules (computer)
├── phone_rule_match.csv                 # Mapping between items and rules (phone)
├── generate_transformation_function_0526_rearrange.csv  # Transformation rule metadata
│
├── Computer - Selected vs. Appeared Counts.pdf          # Exported figure (computer)
├── Computer - Selection Rate by Mode, Level, and Style with 95% Confidence Intervals.pdf
├── Phone - Selected vs. Appeared Counts.pdf             # Exported figure (phone)
└── Phone - Selection Rate by Mode, Level, and Style with 95% Confidence Intervals.pdf

```

> Note: Please keep all CSV files in the same directory as the notebooks (or adjust the file paths inside the notebooks accordingly).

## 2. Software requirements

The analysis was developed and tested with:

- **Python:** 3.12.7
- **Jupyter Notebook**

Required Python packages (imported in the first cell of each notebook):

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `networkx`
- `statsmodels`

Standard-library modules (`collections`, `ast`, `math`, `re`) are also used but do not require separate installation.

### Quick install

From a clean virtual environment:

```bash
pip install \
  pandas \
  numpy \
  matplotlib \
  seaborn \
  networkx \
  statsmodels \
  jupyter
```

(If you already have Jupyter/JupyterLab installed, you can omit `jupyter`.)

## 3. How to run the analysis

1. **Create and activate a Python environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate          # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install pandas numpy matplotlib seaborn networkx statsmodels jupyter
   ```

3. **Launch Jupyter**

   From the directory containing `SurveyDataRound1/`:

   ```bash
   jupyter notebook
   ```

4. **Open and run the notebooks**

   - Open `SurveyDataRound1/ComputerRound1.ipynb`
   - In the menu, choose **Kernel → Restart & Run All**

     (or click “Run All” in your environment)

   This should execute the entire notebook from top to bottom and reproduce the statistics and figures for the **computer** condition.

   Repeat the same steps for:

   - `SurveyDataRound1/PhoneRound1.ipynb`

     to reproduce the analysis for the **smartphone** condition.

No manual parameter tuning is required; all paths are relative and the notebooks assume the CSV files are in the same folder.

## 4. Expected outputs

Running each notebook will:

- Load the corresponding Round 1 survey data from CSV files.
- Compute selection/recognition statistics for each transformation rule.
- Calculate binomial confidence intervals using

  `statsmodels.stats.proportion.proportion_confint`.

- Generate the figures that appear in the paper (e.g., selected vs. appeared counts, selection/recognition rates by mode/level/style).

Some figures are additionally saved as PDF files (those included in this folder).

## 5. Troubleshooting

- **ModuleNotFoundError** (e.g., `No module named 'pandas'`):

  Ensure all packages listed above are installed in the environment where you run Jupyter.

- **FileNotFoundError** for a CSV file:

  Confirm that:

  - You are running the notebook from inside the `SurveyDataRound1` directory, or
  - The CSV files are located in the same directory as the notebook, and
  - The filenames match those listed in Section 1.

If any issues arise that prevent the notebooks from running with “Run All,” they can typically be resolved by checking the working directory and installed packages.

## 6. Limitations

- The artifact includes only the processed survey data; raw Qualtrics exports containing PII cannot be released.
- Visualization styling may vary slightly depending on local Matplotlib defaults.
- Analysis notebooks assume the directory structure remains unchanged.

---

## 7. Ethical and Privacy Considerations

This artifact contains only anonymized survey responses.

All PII has been removed and no re-identifiable information remains.

The analysis procedures comply with the IRB protocol referenced in the paper, including secure storage, anonymization, and aggregated reporting.

---
