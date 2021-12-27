from fastapi import FastAPI
from worker import celery


app = FastAPI()


@app.get("/")
def home_page():
    return {"message": "/docs to see allowed methods"}


@app.post("/predict/{ticker}")
async def predict(ticker: str):
    task_name = "predict.task"
    task = celery.send_task(task_name, args=[ticker])
    response = {
        'id': task.id,
        'url': f'localhost:5000/status/{task.id}/'
    }
    return response


@app.get("/status/{id}")
def check_task(id: str):
    task = celery.AsyncResult(id)
    if task.state == 'SUCCESS':
        response = {
            'task_id': id,
            **task.result
        }
    else:
        response = {
            'status': task.state,
            'result': task.info,
            'task_id': id
        }
    return response
