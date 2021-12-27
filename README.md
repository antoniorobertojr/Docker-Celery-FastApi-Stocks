# Docker-Celery-FastApi-Stocks

This is a Python project to predict if the price over the next 7 trading days will go up or down.
The closer the value is to one, the greater the chance that the price will rise.
The response also returns the train and tests accuracy and precision, validated using a method that respects the time series order.

## Installation

Before running the api, make sure you have docker installed on your machine.

1-Create a folder and Clone the github repository into it

```bash
git clone https://github.com/juniormach96/stocks .
```

2-Build the docker-compose services and run them

```bash
docker-compose up --build
```

## Usage

Access the endpoints through http://localhost:5000/docs

## Future improvements

- Write more tests on celery and api as well
- Test more machine learning algorithms
- Hyperparameter tuning the algos with best scores
