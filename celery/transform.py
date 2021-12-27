import pandas as pd
from featurewiz import featurewiz
import numpy as np
from utils import split_df


def create_return(df):
    """ Creating Return column """

    df['return'] = df['Adj Close'].pct_change()
    return df


def create_variation(df):
    """ Creates a column with variation of the given ticker on the day """

    df['Variation'] = df['Adj Close'].sub(df['Open'])
    return df


def create_moving_average(df, features, windows=[5, 10, 15, 30]):
    """ Make moving average of the selected features """

    df['Date'] = pd.to_datetime(df['Date'])
    # Make a copy to preserve original features
    # And concat them later
    df2 = df.copy()

    num_cols = df[features].select_dtypes('number').columns.to_list()
    num_cols.append('Date')
    features_df = df[num_cols]

    rol = []
    new_features = []

    for col in num_cols:
        if col != 'Date':
            for w in windows:
                features_df[f'avg_last_{w}_{col}'] = features_df[col].rolling(
                    window=w).mean()

                new_features.append(f'avg_last_{w}_{col}')
        else:
            continue

    rol.append(features_df)
    rol_df = pd.concat(rol, ignore_index=True).sort_values(
        'Date').reset_index(drop=True)
    return pd.merge(rol_df, df2)  # .dropna(subset=new_features)


def create_target(df):
    """
    Create a binary column that indicates the direction
    of the closing price on next 7 days
    """

    df['Close_next_7_days'] = df['Adj Close'].transform(lambda x: x.shift(-6))
    df['profit_loss'] = (
        (df['Close_next_7_days'] - df['Open'])/(df['Open']) * 100)
    df['target'] = (df['profit_loss'] > 0).astype(int).shift(-1)
    return df


def winsorization(df, target, lower=5, upper=95):
    """
    Transforms the data by limiting extreme values to selected percentils.
    It uses the function split_df to prevent data leakage.
    """

    # Select binary columns to exlude them
    binary_cols = [col for col in df if np.isin(
        df[col].unique(), [0, 1]).all()]

    # Split on train and test to prevent data leakage
    Xtrain, Xtest, ytrain, ytest = split_df(df, target=target)

    # Select numeric columns except the binary
    features = [
        col for col in Xtrain.select_dtypes('number').columns
        if col not in binary_cols
    ]

    for col in features:
        winsor_lower = np.percentile(Xtrain[col], lower)
        winsor_upper = np.percentile(Xtrain[col], upper)
        Xtrain[col].clip(lower=winsor_lower, upper=winsor_upper, inplace=True)

    train_df = pd.concat([Xtrain, ytrain], axis=1)
    test_df = pd.concat([Xtest, ytest], axis=1)
    print('Made winsorization')
    return pd.concat([train_df, test_df])


def drop_categorical(df):
    return df.select_dtypes(exclude=['object'])


def drop_live_data(df):
    df = df.drop(['return', 'profit_loss', 'Close_next_7_days'], axis=1)
    return df


def separate_predict_data(df):
    """
    Separate data where the predictions will be made.
    We can select the future data selecting the last row,
    since the data is ordered by the date.
    """

    to_predict = df.tail(1)
    df = df.head(-1)
    processed_df = drop_live_data(df)
    # processed_df, to_predict = select_features(0.5, processed_df, to_predict)
    selected_features = ['Variation', 'weekday', 'month', 'target',
                         'avg_last_30_Open', 'avg_last_30_Variation', 'avg_last_30_Volume']
    return processed_df[selected_features], to_predict[selected_features]


def clean_date(df, date_col, drop=True):
    df[date_col] = pd.to_datetime(df[date_col])
    #df = df.sort_values(date_col).reset_index(drop=True)
    df['weekday'] = df[date_col].dt.weekday
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    if drop:
        return df.drop([date_col], axis=1)
    return df


def select_features(corr_limit=0.5, *args):
    processed_df = args[0]
    to_predict = args[1]

    features = featurewiz(
        # Set the dataset you want to do the feature selection
        # Featurewiz is not working with complete dataframe
        # So we are picking a fraction of it
        dataname=processed_df,
        test_data=to_predict,
        # Set the target column
        target='target',
        # Set threshold for removing variables,
        # based on Pearson correlation
        corr_limit=corr_limit,
        verbose=0
    )

    processed_df = features[0]
    to_predict = features[1]

    return processed_df, to_predict


def pipeline(raw_df):
    """
    Reads the raw data of the ticker and process it
    creating the following columns:
    "return, variation, moving_average from some features and target"
    """

    features = [
        'Date', 'High', 'Low', 'Open', 'Volume',
        'Adj Close', 'Symbol', 'Variation']

    processed_df, to_predict = (
        raw_df.
        pipe(create_return).
        pipe(create_variation).
        pipe(create_moving_average, features=features).
        pipe(create_target).
        pipe(
            winsorization,
            target='target',
            lower=5,
            upper=95).
        pipe(drop_categorical).
        pipe(clean_date, date_col='Date').
        pipe(separate_predict_data)
    )

    return processed_df, to_predict
