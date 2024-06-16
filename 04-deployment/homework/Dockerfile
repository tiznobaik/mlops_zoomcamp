FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

# Leverages your existing Pipenv setup instead of 
COPY Pipfile Pipfile.lock ./

RUN pip install pipenv

RUN pipenv install --deploy --ignore-pipfile

# COPY starter.py .
COPY [ "starter.py", "./" ]

CMD ["pipenv", "run", "python", "starter.py", "2023", "05"]