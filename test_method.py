import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from GLMTree.LMTree.method.LMTree import LMTree
import warnings

warnings.filterwarnings("ignore")

dataName = "Australian"  # or "Australian" or "abalone"....
df = pd.read_csv(os.path.join('data', f'{dataName}.csv'))
file_path = os.path.join('data', f'{dataName}.json')

# Load JSON configuration file
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
dataset_name = data['dataset_name']
type = data['type']
num_samples = data['num_samples']
description = data['description']
attribute_introduction = data['attribute_introduction']
target = data['target']
is_categorical = data['is_categorical']

if type:
    taskType = "classification"
else:
    taskType = "regression"
dataName = dataset_name


def Data_sampling(task_type, X_train, y_train, X_test, y_test, target_name, Num_sample=125, Random_status=42):
    """Data sampling function for both classification and regression tasks."""

    def sample_data(data, target_name, num_samples, random_status):
        """Helper function to sample data."""
        sampled_data = data.groupby(target_name).apply(
            lambda x: x.sample(n=1, random_state=random_status)).reset_index(drop=True)

        remaining_samples = num_samples - sampled_data.shape[0]
        if remaining_samples > 0:
            df_sample_remaining = data.sample(n=remaining_samples, random_state=random_status, replace=True)
        sampled_data = pd.concat([sampled_data, df_sample_remaining]).sample(n=num_samples,
                                                                             random_state=random_status).reset_index(
            drop=True)
        return sampled_data

    def data_dropNAN(X_train_sample, y_train_sample, X_test_sample, y_test_sample):
        """Handle missing values by dropping columns with NaN."""
        train_data = pd.concat([X_train_sample, y_train_sample], axis=1)
        test_data = pd.concat([X_test_sample, y_test_sample], axis=1)
        combined_data = pd.concat([train_data, test_data], axis=0)
        combined_data = combined_data.dropna(axis=1)

        train_size = len(train_data)
        X_train = combined_data.iloc[:train_size, :-1]
        y_train = combined_data.iloc[:train_size, -1]
        X_test = combined_data.iloc[train_size:, :-1]
        y_test = combined_data.iloc[train_size:, -1]

        return X_train, y_train, X_test, y_test

    if task_type:  # Classification: stratified sampling
        train_data = X_train.assign(target=y_train)
        train_data.columns.values[-1] = target_name
        X_train_sampled = sample_data(train_data, target_name, Num_sample, Random_status)
        y_train_sampled = X_train_sampled[target_name]
        X_train_sampled = X_train_sampled.drop(columns=[target_name])
        return X_train_sampled, y_train_sampled, X_test, y_test

    else:  # Regression: random sampling
        X_train_num_samples = len(X_train)
        X_train_sample = X_train.sample(n=min(2500, X_train_num_samples), random_state=Random_status)
        y_train_sample = y_train.loc[X_train_sample.index]
        return data_dropNAN(X_train_sample, y_train_sample, X_test, y_test)


def convert_column(col):
    """Convert column to numeric type (int or float)."""
    try:
        return pd.to_numeric(col, errors='raise')
    except ValueError:
        return col


def Experiment_evaluate(ML_model, X_test, y_test, task_type, X, y):
    """Evaluate model performance for both classification and regression tasks."""
    pred = ML_model.predict(X_test)
    if task_type:  # Classification
        acc = accuracy_score(y_test, pred)
        pred_proba = ML_model.predict_proba(X_test)
        unique_classes = np.unique(y)
        if len(unique_classes) > 2:
            auc = roc_auc_score(y_test, pred_proba, multi_class="ovo")
        else:
            auc = roc_auc_score(y_test, pred_proba[:, 1])
        Indicator1, Indicator2 = acc, auc
    else:  # Regression
        r2 = r2_score(y_test, pred)
        y_mean = np.mean(y_test)
        mae = np.sum(np.abs(y_test - pred))
        mae_baseline = np.sum(np.abs(y_test - y_mean))
        relative_mae = 1 - (mae / mae_baseline)
        Indicator1, Indicator2 = r2, relative_mae
    return Indicator1, Indicator2


# Experiment configuration
Random_status_list = 12
train_test_split_size = 0.5
df_loaded = df.copy()

if type:  # Classification: drop missing values
    df_loaded = df_loaded.dropna()
else:  # Regression: fill missing values with 'nan'
    df_loaded = df_loaded.fillna('nan')

seed = Random_status_list
train, test = train_test_split(df_loaded, test_size=train_test_split_size, random_state=seed)
Data = pd.concat([train, test])
Data = Data.sort_index()

# Run LMTree on TrainValData and transform TestData (see run_LMTree.py)
GLMT = LMTree(train, target, dataName, attribute_introduction, is_categorical, taskType=taskType,
              content_desc=description, random_state=seed)
TrainValData2 = GLMT.run()

# Split training/validation sets
X_TrainVal = TrainValData2.drop(columns=[target])
y_TrainVal = TrainValData2[target]
X_train, X_val, y_train, y_val = train_test_split(X_TrainVal, y_TrainVal, test_size=train_test_split_size,
                                                  random_state=seed)

# Apply feature transformation to the test set
X_Test, y_test = GLMT.FeatureTransform(test.drop(columns=[target]), test[target])

# Sample and convert to numeric
X_train, y_train, X_Test, y_test = Data_sampling(type, X_train, y_train, X_Test, y_test, target, Num_sample=125,
                                                 Random_status=seed)
X_train = X_train.apply(convert_column)
X_Test = X_Test.apply(convert_column)

# Initialize ML model
if taskType == "classification":
    ML_model = XGBClassifier()
else:
    ML_model = XGBRegressor()

param_name = "random_state"
if param_name and param_name in ML_model.get_params():
    ML_model.set_params(**{param_name: seed})

# Train and evaluate model
ML_model.fit(X_train, y_train)
Indicator1, Indicator2 = Experiment_evaluate(ML_model, X_Test, y_test, type, X_TrainVal, y_TrainVal)

if type:
    print(f"acc:{Indicator1} auc:{Indicator2}")
else:
    print(f"r^2:{Indicator1} RMAE:{Indicator2}")