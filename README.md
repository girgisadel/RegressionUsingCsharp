# Taxi Fare Prediction with ML.NET

A machine learning project to predict taxi fares using ML.NET. This solution includes end-to-end data preprocessing, training, evaluation, and prediction, designed for both learning and practical deployment.

## Overview

This project leverages ML.NET's FastTree regression algorithm to predict taxi fares based on features such as trip distance, passenger count, and payment type. The implementation extends the official ML.NET sample.

## Features

 - **Data Cleaning & Preprocessing**: Handles missing values, outliers, and normalization.
 - **Custom Reports**: Generate detailed insights like missing values, distinct value counts, and frequency distributions.
 - **Advanced ML Pipeline**: Combines categorical encoding, normalization, and feature engineering for robust predictions.
 - **Save & Reuse Models**: Serialize the trained model for future predictions.

## Getting Started

**Prerequisites**

 - .NET 6 SDK or later.
 - A dataset for taxi fare prediction (example: `taxi-fare-full.csv`).

**Running the Project**

 1. Clone the repository:
```git
git clone https://github.com/girgisadel/RegressionUsingCsharp.git
cd taxi-fare-prediction
```
 2. Build and run the application:
```git
dotnet build
dotnet run
```
 3. View outputs such as data reports, training progress, and evaluation metrics directly in the console.

## Results

The model achieves high accuracy in predicting taxi fares, evaluated against metrics such as:

 - **R² Score**: Measures the proportion of variance captured by the model.
 - **RMSE (Root Mean Squared Error)**: Quantifies prediction error.

## Acknowledgments

This project builds on the [ML.NET Samples](https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/Regression_TaxiFarePrediction).
