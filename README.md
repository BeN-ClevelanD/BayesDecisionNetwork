
## Project Overview

This repository implements a **Bayesian Decision Network (Decision Network AI)** aimed at modeling student performance. It employs probabilistic graphical modeling to capture relationships between variables, compute conditional probability tables (CPTs), and perform both analysis and inference.

### Technologies and Concepts

* **Python** is the primary programming language. Files include both scripts (`.py`) and Jupyter notebooks (`.ipynb`).
* **Bayesian Networks and Decision Networks**: Used for probabilistic modeling of variables and their dependencies.
* **Conditional Probability Tables (CPTs)**: Computed, stored.
* **Data Preprocessing**: Handled in `preprocessdata.py`, preparing raw student datasets (e.g. `student-por.csv`) into a clean format (`processed_port_data.csv`).
* **Model Construction**: `buildBN.py` and `notebookDNBuild.ipynb` define network structure, relationships between variables, and assembly of the decision network; `buildBN.ipynb` offers exploratory and visual analysis within a notebook.
* **CPT Calculation and Experimentation**: Scripts like `expcalccpts.py` and `newcptcalculator.py` support experimentation, adjusting CPTs to evaluate different modeling assumptions.
* **Data Analysis**: `dataAnalysis.py` performs exploratory analysis, statistical summaries.
* **Utility Functions**: Collected in `utility functions.txt` standard routines used across scripts.

### Dataset and Workflow

* **Input Data**: Raw student performance dataset (`student-por.csv`).
* **Preprocessing**: `preprocessdata.py` converts raw data to `processed_port_data.csv`.
* **Model Development**:

  1. **Define Network Structure**: Using notebook (`buildBN.ipynb`) or script (`buildBN.py`).
  2. **Compute CPTs**: Through automated (`BNwthautomcpts.py`) or experimental (`expcalccpts.py`, `newcptcalculator.py`) scripts, with outputs in text files (`cpts_bayesian_network.txt`, `new_cpts_bayesian_network.txt`).
* **Analysis & Evaluation**: Via `dataAnalysis.py`, investigating model behavior, predictive performance, or variable influence.
