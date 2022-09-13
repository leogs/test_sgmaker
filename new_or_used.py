"""
Exercise description
--------------------

Description:
In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the markeplace is new or used.

Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution 
to predict if an item is new or used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k_checked_v3.jsonlines` and a function to read that dataset in `build_dataset`.

For the evaluation, you will use the accuracy metric in order to get a result of 0.86 as minimum. 
Additionally, you will have to choose an appropiate secondary metric and also elaborate an argument on why that metric was chosen.

The deliverables are:
--The file, including all the code needed to define and evaluate a model.
--A document with an explanation on the criteria applied to choose the features, 
  the proposed secondary metric and the performance achieved on that metrics. 
  Optionally, you can deliver an EDA analysis with other formart like .ipynb



"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import (
    train_test_split, RepeatedStratifiedKFold,
    cross_val_score)
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, cohen_kappa_score)
from sklearn.preprocessing import StandardScaler

from utils import eda, transform


# You can safely assume that `build_dataset` is correctly implemented
def build_dataset():
    data = [json.loads(x) for x in open("MLA_100k_checked_v3.jsonlines")]
    target = lambda x: x.get("condition")
    N = -10000
    X_train = data[:N]
    X_test = data[N:]
    y_train = [target(x) for x in X_train]
    y_test = [target(x) for x in X_test]
    for x in X_test:
        del x["condition"]
    return X_train, y_train, X_test, y_test

def data_prune(df: pd.DataFrame,
               dt_cols: list,
               date_prune: str,
              y_col: str) -> pd.DataFrame:
    
    df[dt_cols] = (
        df[dt_cols].apply(pd.to_datetime, errors='coerce'))
    
    df = df[
        df[dt_cols[0]].dt.date >=
        pd.Timestamp(date_prune).date()].reset_index(drop=True)
    
    return df.drop([y_col], axis=1), df[y_col].values

def model_train(X_train: pd.DataFrame, y_train: list):
    model = lgb.LGBMClassifier(n_estimators=500)
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    
    print('\nCross validation mean score: {:.4f} (std: {:.4f})'.format(
        np.mean(scores), np.std(scores)))
    
    model.fit(X_train, y_train)
    
    return model

def model_eval(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    print('\nTest dataset model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
    print('\nCohen Kappa Score: {0:0.4f}'.format(cohen_kappa_score(y_test, y_pred)))
    
    cm = confusion_matrix(y_test, y_pred)
    print('\nConfusion matrix\n\n', cm)
    print('\nTrue Positives(TP) = ', cm[0,0])
    print('True Negatives(TN) = ', cm[1,1])
    print('False Positives(FP) = ', cm[0,1])
    print('False Negatives(FN) = ', cm[1,0])
    
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    print("Loading dataset...")
    # Train and test data following sklearn naming conventions
    # X_train (X_test too) is a list of dicts with information about each item.
    # y_train (y_test too) contains the labels to be predicted (new or used).
    # The label of X_train[i] is y_train[i].
    # The label of X_test[i] is y_test[i].
    X_train, y_train, X_test, y_test = build_dataset()

    # Insert your code below this line:
    
    # Convert train dataset to Pandas DataFrame
    X_train = pd.json_normalize(X_train)
    
    # Remove train data from before August 2015
    X_train, y_train = data_prune(X_train,
                         ['date_created', 'last_updated'],
                         '2015-08-01', 'condition')
    
    X_train, y_train = transform.transform(X_train, y_train, load=False)
    
    scaler = StandardScaler()
    X_train[['base_price',
             'sold_quantity',
             'variations_count',
             'diff_start_stop_time',
             'create_update_diff_days']] = scaler.fit_transform(X_train[['base_price',
                                                                         'sold_quantity',
                                                                         'variations_count',
                                                                         'diff_start_stop_time',
                                                                         'create_update_diff_days']])
    X_test, y_test = transform.transform(X_test, y_test)

    X_test[['base_price',
             'sold_quantity',
             'variations_count',
             'diff_start_stop_time',
             'create_update_diff_days']] = scaler.transform(X_test[['base_price',
                                                                         'sold_quantity',
                                                                         'variations_count',
                                                                         'diff_start_stop_time',
                                                                         'create_update_diff_days']])
    
    print("\nTraining model...")
    
    model = model_train(X_train, y_train)

    print("\nModel evaluation...")
    
    model_eval(model, X_test, y_test)