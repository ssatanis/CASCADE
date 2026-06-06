FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .
# The benchmark reproduces from committed result JSONs — no data download, no retrain.
CMD ["cascade", "benchmark", "--leaderboard"]
