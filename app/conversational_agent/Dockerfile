FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy proto and generate Python stubs
COPY proto/ proto/
RUN python -m grpc_tools.protoc \
    -Iproto \
    --python_out=. \
    --grpc_python_out=. \
    proto/audio.proto

COPY server.py .

RUN mkdir -p /data/audio

EXPOSE 50051

CMD ["python", "server.py"]
