FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential make libopenmpi-dev openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home heat
WORKDIR /home/heat/app
COPY . .
RUN chown -R heat:heat /home/heat/app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e ".[dashboard]"

USER heat
RUN make build CFLAGS="-O3 -std=c11"

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "src/heatlab/dashboard/app.py", "--server.address=0.0.0.0"]
