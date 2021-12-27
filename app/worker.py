import os
from celery import Celery

CELERY_BROKER_URL = os.getenv("REDISSERVER")
CELERY_RESULT_BACKEND = os.getenv("REDISSERVER")

celery = Celery("celery", backend=CELERY_BROKER_URL,
                broker=CELERY_RESULT_BACKEND)
