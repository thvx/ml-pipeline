FROM python:3.11.4

WORKDIR /app

# Copia el archivo de requerimientos y la aplicación
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expone el puerto en el que correrá la API
EXPOSE 5000

# Comando para correr la aplicación
CMD ["python", "app/api.py"]
