FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY business_entity_resolution/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY business_entity_resolution/src /app/business_entity_resolution/src
COPY dataset /app/dataset
COPY utils /app/utils
RUN mkdir -p /app/output
CMD ["python", "business_entity_resolution/src/entity_resolution.py", "--mode", "test"]
