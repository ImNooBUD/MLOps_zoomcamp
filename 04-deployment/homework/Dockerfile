FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

WORKDIR /app
COPY [ "starter.py", "starter.py" ]
COPY [ "requirements.txt", "requirements.txt" ]

RUN pip install -r requirements.txt
RUN python starter.py year=2021 month=4