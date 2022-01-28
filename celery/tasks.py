import extract
import transform
import predict
import traceback
from celery import states
from worker import celery


@celery.task(name='predict.task', bind=True)
def main(self, ticker):
    """
    Predict wheter the price will goes up or down
    and give the train and tests accuracy
    """
    try:
        # Extract
        raw_df = extract.get_data(ticker)
        # Transform
        processed_df, to_predict = transform.pipeline(raw_df)
        # Predict
        predictions = predict.pipeline(processed_df, to_predict)
        predictions['ticker'] = ticker
        return predictions
    except Exception as ex:
        self.update_state(
            state=states.FAILURE,
            meta={
                'exc_type': type(ex).__name__,
                'exc_message': traceback.format_exc().split('\n')
            })
        raise ex
