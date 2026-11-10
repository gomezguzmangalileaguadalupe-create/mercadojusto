# Mercado Justo — despliegue en Render

Aplicación web en Python + Gradio, con PySpark, Random Forest, Plotly, Folium y NetworkX.

## Archivos necesarios

Coloca estos archivos en la raíz del repositorio:

- `mercado.py`
- `base_carne_14000_datos.xlsx`  **(obligatorio; no está incluido en este paquete)**
- `requirements.txt`
- `Dockerfile`

## Desplegar en Render

1. Sube el contenido de esta carpeta a un repositorio de GitHub.
2. Entra a https://render.com/ e inicia sesión.
3. Selecciona **New +** → **Web Service**.
4. Conecta tu repositorio de GitHub.
5. En Runtime/Environment selecciona **Docker**.
6. Deja `Dockerfile` como ruta del archivo Docker.
7. Crea el servicio y espera a que termine el build.
8. Cuando el despliegue termine, abre la URL `onrender.com` que Render proporciona.

No necesitas escribir un Build Command ni un Start Command: Dockerfile ya contiene la instalación y el comando de inicio.

## Datos de Excel

El archivo `base_carne_14000_datos.xlsx` debe estar en la misma carpeta que `mercado.py` y subirse al repositorio junto con los demás archivos.

El Excel debe contener al menos estas columnas:

`fecha`, `estado`, `producto`, `precio_kg`, `mercado`, `tipo_alimentacion`, `inflacion`, `temperatura`.

Si el archivo tiene otro nombre o está en otra ruta, configura la variable de entorno `DATA_FILE` en Render con la ruta correspondiente.

## Configuración de Render

- Tipo de servicio: Web Service
- Runtime: Docker
- Dockerfile Path: `Dockerfile`
- Puerto HTTP: la aplicación lee la variable `PORT` de Render; si no existe, usa `10000`.
- Región: elige una región cercana a tus usuarios.

## Recursos recomendados

PySpark inicia una JVM además del proceso de Python. Para evitar errores de memoria, utiliza un plan de Render con memoria suficiente; como punto de partida, considera al menos 2 GB y preferentemente 4 GB para pruebas más estables. El tamaño necesario depende de la carga y de los datos.

## Notas importantes

- El sistema conserva las cuatro fases del dashboard.
- El modelo Random Forest se entrena al iniciar la aplicación.
- La carga inicial de Spark y el entrenamiento pueden hacer que el primer arranque tarde.
- En servicios gratuitos, Render puede suspender la instancia tras un periodo sin tráfico; el primer acceso posterior puede tardar en responder.
- El sistema no requiere GPU para el Random Forest.
- El sistema de archivos de Render es efímero por defecto. Como el Excel se incluye en la imagen al desplegar, estará disponible en cada despliegue. No guardes archivos subidos por usuarios localmente si necesitas conservarlos.

## Ejecución local con Docker

Desde esta carpeta, después de agregar el Excel:

```bash
docker build -t mercado-justo .
docker run --rm -p 10000:10000 mercado-justo
```

Abre `http://localhost:10000`.

## Solución de problemas

- **No se encontró el Excel:** revisa que el nombre sea exactamente `base_carne_14000_datos.xlsx` y que esté en la raíz.
- **Java gateway process exited:** revisa que el contenedor use el Dockerfile incluido y que Java 17 se haya instalado correctamente.
- **Out of memory:** aumenta la memoria del servicio de Render.
- **Port scan timeout:** verifica que el servicio esté iniciando `mercado.py` y que Gradio escuche en `0.0.0.0` usando `PORT`.
