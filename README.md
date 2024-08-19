# Project Description

## Overview

The objective of this MLOps project is to build a robust machine learning pipeline for predicting the success of marketing campaigns using data sourced from Kaggle. This pipeline will incorporate experiment tracking, orchestration, deployment, and monitoring to ensure a scalable and reliable solution.

## Data
The project utilizes a dataset from Kaggle that includes the following features: 'GoalAmount', 'RaisedAmount', 'DurationDays', 'NumBackers', 'Category', 'LaunchMonth', 'Country', 'Currency', 'OwnerExperience', 'VideoIncluded', 'SocialMediaPresence', 'NumUpdates', 'IsSuccessful'. The dataset is preprocessed and split into training and testing sets to facilitate model development and evaluation. 

Here is the link to the dataset - [campaign data](https://www.kaggle.com/datasets/preethamgouda/campaign-data)


## Methodology

1. **Exploratory Data Analysis (EDA)**:
   - Performed exploratory data analysis on the campaign dataset to gain insights into the data distribution, identify potential issues (e.g., missing values, outliers), and understand the relationships between features and the target variable
   - Visualized the data using plots like violin plot

2. **Data Preprocessing**:
   - Encoded categorical variables using techniques like one-hot encoding or label encoding
   - Split the preprocessed data into training, validating and testing sets

3. **Model Selection and Training**:
   - Chose Random Forest Classification as the modeling technique due to its ability to handle non-linear relationships, robustness to outliers, and ease of interpretation
   - Trained the Random Forest model on the training data and evaluated its performance on the validating data using appropriate metrics (e.g., accuracy, precision, recall, auc scores)
   - Performed hyperparameter tuning and inspected the results with MLflow

4. **Model Evaluation and Selection**:
   - Compared the various results with various parameters in MLflow and chose the best parameters
   - Combined training dataset and validation dataset and trained again for the final model
   - Stored and logged final model in MLflow artifact 


**[Link](notebook.ipynb)**

## Experiment Tracking

**Tools**: [MLflow](https://mlflow.org/)

**Description**: Implement a system to log and track experiments, including hyperparameters, model versions, training metrics, and evaluation results. This enables reproducibility, comparison of different model configurations, and easy retrieval of past experiments.

**Environment Creation**

```
conda create -n campaign-data-prediction
```

```
conda activate campaign-data-prediction
```

```
pip install -r requirements.txt
```
<br><br>
<image src="images/comparing_100_runs.png" alt="compare_hyperparameters">

*Fig: Comparing Hyperparameter Tuning Results for max_depth, min_samples_leaf, and n_estimators*

<br>
<image src="images/sorting_models_with_roc_auc.png" alt="compare_hyperparameters">

*Fig: Comparing Hyperparameter Tuning Results for max_depth, min_samples_leaf, and n_estimators*

**[Link](notebook.ipynb)**

## Orchestration

**Tools**: [Mage](https://www.mage.ai/)

**Description**: Set up an orchestration framework to automate the end-to-end machine learning workflow. This includes data ingestion, preprocessing, model training, and model logging. The orchestration system ensures that each step of the pipeline is executed in the correct order and manages dependencies between tasks.
<br><br>

<image src="images/orchestration.png" alt="orchestration pipeline">

*Fig: Orchestration pipeline of the workflow*

**[Link](orchestration)**

## Deployment

**Tools**: [Docker](https://www.docker.com/)

**Description**: Build and deploy the machine learning model as a batch prediction service and a web service using Docker containers. The batch prediction service allows for processing large datasets offline, while the web service exposes an API for making real-time predictions.

**Batch**

```
cd .\deployment\batch_deployment\ 
```

```
docker build -t campaign-success-prediction .
```
<br>
Docker container can be tested with the following code by inserting parquet file link of the data. 
<br><br>


```
docker run campaign-success-prediction python predict.py https://github.com/Hokfu/campaign_success_prediction_with_mlops/raw/master/campaign_data.parquet
```

**Web Service**

```
cd .\deployment\web_service_deployment\ 
```

```
docker build -t campaign-success-prediction-web .
```

```
docker run -it --rm -p 8000:8000 campaign-success-predictios-prediction-web
```


**[Link](deployment)**


## Monitoring

**Tools**: [Evidently](https://www.evidentlyai.com/), [Grafana](https://grafana.com/)

**Description**: The monitoring system tracks the following metrics:

- Column Drift for the prediction column: Monitors the distribution of the prediction column to detect any drift from the reference data.
- Dataset Drift: Monitors the overall drift of the dataset compared to the reference data, considering all numerical and categorical features.
- Dataset Missing Values: Tracks the percentage of missing values in the dataset.

These metrics are calculated using the Evidently library and visualized in Grafana dashboards for easy monitoring and analysis.

<image src="images/avg_prediction_drift.png" alt="average_prediction_drift">

*Fig: Average Prediction Column Drift*

The prediction drift metric measures the distribution shift between the prediction column in the reference data and the current data. A drift score of 0 indicates no drift, while a higher score signifies a larger deviation from the reference distribution.

In the monitoring system, the average prediction drift is observed to be stable at around 0.00349. This low drift score suggests that the distribution of predictions in the current data remains consistent with the reference data used for training the model. A stable prediction drift is desirable as it indicates that the model's predictions are not deviating significantly from the expected behavior, ensuring reliable performance over time.

**[Link](monitoring)**

## Best Practices

#### Model Prediction Testing

To ensure the reliability of the model's predictions, a set of tests are implemented using the pytest framework. 

**[Link](best-practices)**

