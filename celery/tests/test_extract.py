# use "python -m unittest discover"
# to call the functions

import extract
import unittest
import datetime as dt


class TestPreProcessing(unittest.TestCase):

    def setUp(self):
        self.df = extract.get_data(ticker='BTC-USD')

    def test_get_data_response(self):
        # Some tickers don't have data since 2015-01-01
        self.assertEqual(self.df.Date.max().day, dt.datetime.now().day)
        self.assertEqual(self.df.Date.max().month, dt.datetime.now().month)
        self.assertEqual(self.df.Date.max().year, dt.datetime.now().year)
