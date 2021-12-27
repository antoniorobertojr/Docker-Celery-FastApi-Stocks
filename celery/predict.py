import pandas as pd
from utils import split_df
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from collections import defaultdict
from lightgbm import LGBMClassifier


def validation(df, features, target, models, n_folds=5):
    # Split data in train and test to prevent data leakage
    Xtrain, _, ytrain, _ = split_df(df, target=target)
    print(f'Training Columns {Xtrain.columns}')
    # Create a train dataframe
    df = pd.concat([Xtrain, ytrain], axis=1)
    window = int(df.shape[0]/(n_folds+1))
    result = defaultdict(list)
    for name, model in models:
        # Prequential Sliding
        for n in range(n_folds):
            cut_off_train = window*(n+1)
            cut_off_test = window*(n+2)
            # print(cut_off_train)
            # print(cut_off_test)
            train = df.iloc[window*n:cut_off_train]
            if n != (n_folds):
                val = df.iloc[cut_off_train:cut_off_test]
            else:
                val = df.iloc[cut_off_train:]

            Xtrain = train[features]
            Xval = val[features]
            ytrain = train[target]
            yval = val[target]

            sliding_model = model.fit(Xtrain, ytrain)
            p = sliding_model.predict(Xval)
            Xval['p'] = sliding_model.predict_proba(Xval)[:, 1]

            accuracy = accuracy_score(yval, p)
            precision = precision_score(yval, p)
            roc_auc = roc_auc_score(yval, p)

            result['name'].append(name)
            result['accuracy'].append(accuracy)
            result['precision'].append(precision)
            result['roc_auc'].append(roc_auc)

            # Prequential Expanding
            train = df.iloc[:cut_off_train]
            if n == 0:  # First Iteration of prequential expanding is equal to prequential sliding
                continue
            elif n != (n_folds-1):
                val = df.iloc[cut_off_train:cut_off_test]
            else:
                val = df.iloc[cut_off_train:]

            Xtrain = train[features]
            Xval = val[features]
            ytrain = train[target]
            yval = val[target]

            expanding_model = model.fit(Xtrain, ytrain)
            p = expanding_model.predict(Xval)
            Xval['p'] = expanding_model.predict_proba(Xval)[:, 1]

            accuracy = accuracy_score(yval, p)
            precision = precision_score(yval, p)
            roc_auc = roc_auc_score(yval, p)

            result['name'].append(name)
            result['accuracy'].append(accuracy)
            result['precision'].append(precision)
            result['roc_auc'].append(roc_auc)

    results_df = pd.DataFrame(result).groupby('name').agg(
        {
            'accuracy': ['mean', 'min', 'max'],
            'precision': ['mean', 'min', 'max'],
            'roc_auc': ['mean', 'min', 'max']
        }
    )
    return results_df


def pipeline(*args):
    processed_df = args[0]
    to_predict = args[1]

    Xtrain, Xtest, ytrain, ytest = split_df(processed_df, target='target')
    lgbm_model = LGBMClassifier()
    lgbm_model.fit(Xtrain, ytrain)
    lgbm_model_tuple = [('LGBM', LGBMClassifier())]
    predictions = lgbm_model.predict_proba(to_predict[Xtrain.columns])
    results_df = validation(
        processed_df,
        features=Xtrain.columns,
        target='target',
        models=lgbm_model_tuple,
        n_folds=5
    )
    train_accuracy = results_df['accuracy']['mean']
    train_precision = results_df['precision']['mean']
    test_preds = lgbm_model.predict(Xtest[Xtrain.columns])
    tests_accuracy = accuracy_score(test_preds, ytest)
    tests_precision = precision_score(test_preds, ytest)
    response = {
        'prediction': predictions[0, 0],
        'train_accuracy': np.float(train_accuracy.values),
        'train_precision': np.float(train_precision.values),
        'tests_accuracy': tests_accuracy,
        'tests_precision': tests_precision
    }
    print(response)
    return response
