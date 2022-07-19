FROM python:3.10

WORKDIR /code
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /code
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "api.app:app", "-b", "0.0.0.0:8000"]
