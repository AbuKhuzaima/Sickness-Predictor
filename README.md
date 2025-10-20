# Sickness Prediction System

A machine learning-based web application that predicts diseases based on input symptoms using Support Vector Classification (SVC).

## Overview

This Flask web application uses a machine learning model to predict sickness from text descriptions of symptoms. The system utilizes TF-IDF (Term Frequency-Inverse Document Frequency) vectorization and Support Vector Classification for prediction.

## Features

- Text-based symptom input
- Sickness prediction based on symptoms
- Mean Squared Error (MSE) calculation endpoint
- Simple and intuitive web interface
- Bootstrap-styled frontend

## Technical Stack

- **Backend**: Python/Flask
- **Machine Learning**: scikit-learn
  - TF-IDF Vectorizer
  - Support Vector Classification (SVC)
- **Frontend**: HTML, Bootstrap
- **Data**: CSV-based dataset

## Project Structure

```
.
├── app.py              # Main Flask application
├── disease.csv         # Dataset containing symptoms and diseases
└── templates/
    ├── index.html     # Main input page
    └── result.html    # Prediction result page
```

## Setup and Running

1. Install required dependencies:
```bash
pip install flask pandas scikit-learn
```

2. Run the application:
```bash
python app.py
```

3. Access the application at `http://localhost:5000`

## Usage

1. Open the application in your web browser
2. Enter symptoms in the text area
3. Click "Predict" to get the disease prediction
4. Access `/calculate_mse` endpoint to view model performance

## Model Details

The system uses a machine learning pipeline that consists of:
- TF-IDF Vectorization for text processing
- Support Vector Classification for disease prediction
- 80-20 train-test split for model evaluation
