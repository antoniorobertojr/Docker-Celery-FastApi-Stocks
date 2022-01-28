from pycaret.classification import *


def pipeline(*args):
    processed_df = args[0]
    to_predict = args[1]
    # Split into
    n = int(processed_df.shape[0]*0.75)
    #Train and Test
    train = processed_df.iloc[:n]
    test = processed_df.iloc[n:]

    # Start experiment
    setup(
        train,
        test_data=test,
        target="target",
        silent=True,
        fold_strategy='timeseries',
        data_split_stratify=True,
        verbose=False)

    # Create lgbm model
    lgbm = create_model('lightgbm')
    # Tune it
    tuned_best = tune_model(
        lgbm, search_library='optuna', search_algorithm='tpe')
    # Adjust probabilities
    calibrated = calibrate_model(tuned_best)
    # Train on the whole dataset
    final_model = finalize_model(calibrated)
    # Predict
    result = predict_model(final_model, data=to_predict)
    result = result.fillna('')
    # Change label to became more user-friendly
    result['Label'] = np.where(
        result['Label'] == '0.0',
        'price will go DOWN by the end of next 7 trading days',
        'price will go UP by the end of next 7 trading days')
    # Calculate model score
    model_score = pull()
    # Add it to accuracy
    result['Model Accuracy'] = model_score['Accuracy']['Mean']
    response = result[['Label', 'Score',
                       'Model Accuracy']].to_dict('list')
    return response
