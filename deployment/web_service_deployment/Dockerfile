FROM python:3.12.4-slim-bullseye

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

COPY ["models/rf_clf.bin", "models/dv.bin", "./models/"]

RUN pipenv install --system --deploy

COPY "predict.py" .

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]