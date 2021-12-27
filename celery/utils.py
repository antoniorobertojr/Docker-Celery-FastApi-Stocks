def split_df(df, target, size=0.75):
    """
    Splits data between past and future based on
    index to prevent data leakage on a time series

    Args:
        df: DataFrame to split
        target: column that will be predicted
        size: proportion of train data

    Returns:
        Xtrain: train features
        Xtest: test features
        ytrain: train target
        ytest: test target
    """

    n = int(df.shape[0]*size)
    #Train and Test
    train = df.iloc[:n]
    test = df.iloc[n:]
    #Xtrain and Xtest
    Xtrain = train.drop(target, axis=1)
    ytrain = train[[target]]
    Xtest = test.drop(target, axis=1)
    ytest = test[[target]]
    return Xtrain, Xtest, ytrain, ytest
