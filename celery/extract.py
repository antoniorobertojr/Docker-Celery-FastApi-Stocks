import pandas_datareader.data as web
import datetime as dt


def get_data(ticker):
    """ Get historical data from selected ticker since 2015-01-01. """

    start = dt.datetime(2015, 1, 1)
    end = dt.datetime.now()
    df = web.DataReader(ticker, 'yahoo', start, end)
    df.reset_index(inplace=True)
    df['Symbol'] = ticker
    return df


if __name__ == '__main__':
    df = get_data('BTC-USD')
    print(df.Date.max())
