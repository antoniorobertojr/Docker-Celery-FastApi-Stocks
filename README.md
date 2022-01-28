# Docker-Celery-FastApi-Stocks

This is a Python project to predict if the price over the next 7 trading days will go up or down. It takes the stock ticker as an input, passes the task to the Celery worker through a Redis server and returns an id for the user to check the task state and result. On the backend, it gets data from Yahoo finances, process it, trains a LGBM model and makes a prediction of the price direction over the next 7 trading days. All the services (FastAPI, Celery and Redis) are build through a Docker Compose file.

# Table of Contents

1. [Installation](#example)
2. [Usage](#installation)
3. [Technologies](#technologies)
4. [Future Improvements](#future-improvements)

## Installation

Before running the API, make sure you have [Docker](https://docs.docker.com/get-docker/) installed on your machine.

1. Create a folder and clone the github repository into it

```bash
git clone https://github.com/juniormach96/stocks .
```

2. Build the Docker Compose services and run them

```bash
docker-compose up --build
```

## Usage

1. Open the browser and go to http://localhost:5000/docs

   ![](https://github.com/juniormach96/Docker-Celery-FastApi-Stocks/blob/pycaret/assets/img/fast-api-1.PNG)
2. Click on /predict/{ticker} endpoint and then on 'try it out'

   ![](https://github.com/juniormach96/Docker-Celery-FastApi-Stocks/blob/pycaret/assets/img/fast-api-2.PNG)
3. Fill the blank field with a valid yahoo ticker and click on execute

   ![](https://github.com/juniormach96/Docker-Celery-FastApi-Stocks/blob/pycaret/assets/img/fast-api-3.1.PNG)
4. Copy the url provided by the response and access it

   ![](https://github.com/juniormach96/Docker-Celery-FastApi-Stocks/blob/pycaret/assets/img/fast-api-4.2.PNG)
5. Now you can see the task status. If it has not yet been processed, it will be 'PENDING', otherwise it will return a response indication the price direction for the next 7 days, the likelihood of it, and the model accuracy aswell.

   ![](https://github.com/juniormach96/Docker-Celery-FastApi-Stocks/blob/pycaret/assets/img/fast-api-5.PNG)

   ![](https://github.com/juniormach96/Docker-Celery-FastApi-Stocks/blob/pycaret/assets/img/fastapi-6.PNG)

## Technologies

* Yahoo Finances API: extract the data;
* Pandas: clean and process it;
* Pycaret: create a machine learning model;
* FastAPI: create an endpoint to trigger the tasks;
* Celery: delay the tasks;
* Redis: carry the state and response of them;
* Docker Compose: build the services.

## Future improvements

- Write more tests for the API;
- Allow the user to select how many trading days he wants to work with;
- Add project structure to README.
