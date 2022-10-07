FROM python:slim

WORKDIR /app

RUN apt-get update && apt-get -y install cmake protobuf-compiler

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python3", "detect_from_webcam.py"]