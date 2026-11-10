
FROM python:3.11-slim-bookworm

WORKDIR /app

# Instalar Java para PySpark
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

# Instalar dependencias
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Comprobar PySpark
RUN python -c "import pyspark; print('PySpark:', pyspark.__version__)"

# Copiar aplicación
COPY . .

CMD ["python", "mercado.py"]
