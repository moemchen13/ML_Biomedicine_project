# ML Biomedicine Project

A Streamlit-based web app for experimenting with machine-learning models on biomedical data (tabular + image).

<p align="left">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-3776AB">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-app-red">
</p>

---

## Features

- **Two app modes**
  - **Tabular ML**: load a CSV, pick a pre-trained or freshly trained model, run predictions, inspect metrics.
  - **Image ML**: upload images, run inference with a trained model, and preview results.
- **Pretrained models** stored in `saved_models/` (`*.pkl`) for instant demos.
- **Config-driven** runs via `configuration*.json` (swap datasets, models, and UI defaults without code edits).
- **Example dataset** (`stroke.csv`) and a **playground notebook** for experiments.
- **One-file Streamlit apps** for quick starts:
  - `ML_application_tabular.py`
  - `ML_application_image.py`

> This project is for learning and prototyping. **Do not** use it to make clinical decisions.

---

## Getting Started

### 1) Clone & create a virtual environment
```bash
git clone https://github.com/moemchen13/ML_Biomedicine_project.git
cd ML_Biomedicine_project

# Create + activate a venv (Linux/Mac)
python -m venv .venv
source .venv/bin/activate

# (Windows)
# python -m venv .venv
# .venv\Scripts\activate
```

## Installation

pip install -r requirements.txt

## Run the Application
```
# Tabular demo
streamlit run ML_application_tabular.py

# Image demo
streamlit run ML_application_image.py
```

## Project Structure

ML_Biomedicine_project/
├─ ML_application_tabular.py      # Streamlit app: tabular workflows
├─ ML_application_image.py        # Streamlit app: image workflows
├─ backend_ML.py                  # Reusable ML helpers / training / inference
├─ saved_models/                  # Pretrained models (*.pkl)
├─ data/                          # Demo data (e.g., stroke.csv)
│  └─ pics/                       # (Optional) images for docs/demos
├─ configuration.json             # Default app configuration
├─ configuration_classification.json
├─ configuration_demo.json
├─ playground.ipynb               # Scratchpad for experiments
├─ requirements.txt               # Python dependencies
├─ LICENSE                        # Apache 2.0
└─ .devcontainer/                 # VS Code Dev Container setup

## Using Pretrained Models

- Put pickled estimators in `saved_models/` (e.g., `logreg_trained.pkl`, `dectree_trained.pkl`).
- In the app, select the model in the sidebar (if enabled) **or** point your config to the file.
- Keep names clear and versioned (e.g., `model_v1_dt.pkl`, `model_v2_lr.pkl`) for reproducibility.

**Save a model example:**
```python
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)

with open("saved_models/logreg_trained.pkl", "wb") as f:
    pickle.dump(pipe, f)
```

## Data

- Example dataset: `data/stroke.csv` for quick tabular demos.
- Bring your own CSV:
  - Ensure a **target** column (for supervised learning) and correct **feature** column names.
  - Handle missing values and categorical encoding **consistently** between training and inference.
  - Remove any PII/PHI — this project is for learning only and **not** for clinical use.
  - Keep train/validation/test splits reproducible (e.g., fixed `random_state`).
- For images:
  - Use a directory-per-class structure if doing classification (e.g., `data/images/<class_name>/*`).
  - Keep image sizes/formats consistent or add preprocessing in the pipeline.
  - Update the configuration to point to your image directory.

## Adding a New Model

1. **Train your model** using the same preprocessing you’ll use at inference time.
2. **Save the artifact** to `saved_models/your_model.pkl` (prefer a `Pipeline` so preprocessing is bundled).
3. **Update configuration** to reference the new model and columns (e.g., `configuration.json`).
4. **(Optional) Expose UI controls** in the Streamlit sidebar for model selection or key hyperparameters.

## Have Fun with this library
