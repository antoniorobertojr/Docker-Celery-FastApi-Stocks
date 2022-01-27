from pycaret.classification import *


def pipeline(*args):
    processed_df = args[0]
    to_predict = args[1]
    print(processed_df.columns)
    print(to_predict.columns)
    # Split into
    n = int(processed_df.shape[0]*0.75)
    #Train and Test
    train = processed_df.iloc[:n]
    test = processed_df.iloc[n:]

    # Start experiment
    setup(
        train,
        preprocess=False,
        test_data=test,
        target="target",
        silent=True,
        fold_strategy='timeseries',
        data_split_stratify=True)

    # Create lgbm model
    lgbm = create_model('lightgbm')
    # Tune it
    # tuned_best = tune_model(
    #    lgbm, search_library='optuna', search_algorithm='tpe')
    # Adjust probabilities
    #calibrated = calibrate_model(tuned_best)
    # Train on the whole dataset
    #final_model = finalize_model(calibrated)
    # Predict
    print(train.columns)
    print(to_predict.columns)
    result = predict_model(lgbm, data=test)
    result = result.fillna('')
    # Calculate model score
    features = processed_df.drop('target', axis=1)
    target = processed_df['target']
    model_score = np.round(lgbm.score(X=features, y=target), 3)
    result['Model score'] = model_score
    response = result[['Label', 'Score', 'Model score']].to_dict('records')
    print(response)
    return response
