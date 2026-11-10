
FROM python:3.11-slim-bookworm

WORKDIR /app

# Instalar Java 17
RUN apt-get update && \
    apt-get install -y openjdk-17-jre-headless ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Configurar Java
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar proyecto
COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "mercado.py"]
