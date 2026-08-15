# Pipeline-2 classifier service.
# Deployment-agnostic: the platform (Render/Streamlit/HF/etc.) is chosen later.
# Model weights: mount or COPY the checkpoint dir into /app/models.
# Set AUTH_TOKEN at deploy time.

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# system deps liteparse/opencv need
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
# Placeholder dir for the checkpoint. Platform adds the weights here, e.g. a
# volume/secret mount, or you uncomment the COPY below and provide the files.
RUN mkdir -p /app/models

# To bake weights into the image (public repo only — avoid for confidential):
# COPY models ./models

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]