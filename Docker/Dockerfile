FROM python:3.9-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY "requirements.txt" "./"

RUN uv pip install --system -r requirements.txt

COPY "main.py" "xception_06_0.720.h5" "./"

EXPOSE 5000

ENTRYPOINT ["python", "main.py"]