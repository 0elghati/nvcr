FROM omarelghati/nvcr@sha256:e8b04d3efc3c4048167aef1bc9ff1c72c3b748f8f53584650f0cad761e2a57e9
# Offline smoke quality checks need NumPy; the NVCR binary and engines are unchanged.
RUN apt-get update && apt-get install -y --no-install-recommends python3-numpy && rm -rf /var/lib/apt/lists/*
