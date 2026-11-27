FROM python:3.11-slim

WORKDIR /app

COPY tarpit/classifier.py /app/classifier.py
COPY tarpit/server.py /app/server.py

EXPOSE 8080

CMD ["python", "/app/server.py"]