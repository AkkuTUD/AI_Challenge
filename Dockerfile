FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

EXPOSE 8080

CMD ["python3", "app.py"]