# Student Performance Prediction System

This project is a production-ready, multi-page Streamlit web app that predicts student GPA and provides interactive analytics. The notebook is read-only reference material; all app logic lives in Python modules.

## Highlights

- Dark, student-themed interface with custom navigation
- Interactive charts (Plotly) and diagnostics
- Single and batch GPA prediction
- Fast loading using sample-first data strategy

## Pages and What They Do

- Home: KPIs and project summary
- Dataset Overview: preview, missing data heatmap, correlation, filters, downloads
- Data Analysis: relationships and category-wise performance
- Prediction: single student GPA prediction
- Model Info: model details, metrics, residuals, feature importance, insights
- Batch Prediction: CSV upload and bulk scoring

## Data Sources

The app looks for data in this order:

1. data/StudentsPerformance.csv (single combined file)
2. datasets/train.csv, datasets/test.csv, datasets/validation.csv (split files)

If data/StudentsPerformance.csv exists it becomes the primary source and is also used when the full-data toggle is enabled in Dataset Overview.

## Dataset Schema

Columns in the dataset:

- Age
- Grade
- Gender
- Race
- SES_Quartile
- ParentalEducation
- SchoolType
- Locale
- TestScore_Math
- TestScore_Reading
- TestScore_Science
- GPA
- AttendanceRate
- StudyHours
- InternetAccess
- Extracurricular
- PartTimeJob
- ParentSupport
- Romantic
- FreeTime
- GoOut

Target column:

- GPA

Batch Prediction requires all feature columns except GPA. The categories must match the training data (unseen categories are rejected).

## Preprocessing and Model

The app mirrors the notebook pipeline:

- Gender mapping: Female -> 0, Male -> 1
- SchoolType mapping: Public -> 0, Private -> 1
- Label encoding: Race, ParentalEducation, Locale
- StandardScaler fitted on training data

Model and scaler loading:

- Model: model/model.pkl or student_performance_best_model.pkl
- Scaler: model/scaler.pkl or student_performance_scaler.pkl

## Performance and Sampling

To avoid large payloads and slow load times, the app loads a sample by default:

- Default sample size: 20,000 rows (utils/data_loader.py: DEFAULT_SAMPLE_ROWS)
- Dataset Overview offers a "Use full dataset (slow)" toggle
- Downloads are capped to avoid client message size limits

Current caps:

- Dataset preview: first 200 rows
- Filtered display: up to 2,000 rows
- Download CSV: up to 20,000 rows

## Project Structure

```text
Data Mining Project/
├── app.py
├── README.md
├── requirements.txt
├── runtime.txt
├── Student Performance Prediction.ipynb
├── student_performance_best_model.pkl
├── student_performance_scaler.pkl
├── .streamlit/
│   └── config.toml
├── data/
├── datasets/
│   ├── train.csv
│   ├── test.csv
│   └── validation.csv
├── model/
├── utils/
│   ├── data_loader.py
│   ├── model_loader.py
│   ├── preprocessing.py
│   └── helpers.py
└── pages/
   ├── __init__.py
   ├── home.py
   ├── dataset_overview.py
   ├── data_analysis.py
   ├── prediction.py
   ├── model_info.py
   └── batch_prediction.py
```

Local/generated folders (not shown): venv/, __pycache__/, .git/

## Local Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the app:
   ```bash
   streamlit run app.py
   ```

## Streamlit Cloud Deployment

1. Push the repo to GitHub.
2. Create a new Streamlit app and select app.py.
3. Ensure model and scaler artifacts are included.

## Configuration

- Theme settings: .streamlit/config.toml
- Python version pin: runtime.txt (used by Streamlit Cloud)
- Sample size: utils/data_loader.py (DEFAULT_SAMPLE_ROWS)
- If you must allow larger client payloads, set server.maxMessageSize in Streamlit config. This can increase memory usage and load times.

## Troubleshooting

- "Dataset not available": ensure data/StudentsPerformance.csv or datasets/*.csv exist.
- "Model loading failed": ensure model/model.pkl or student_performance_best_model.pkl exists.
- "Unseen categories": batch input contains categorical values not seen in training data.
- Slow pages or MessageSizeError: lower sample sizes or keep full dataset toggle off.

## Notebook Reference

Student Performance Prediction.ipynb is read-only reference material and must not be modified by the app.
