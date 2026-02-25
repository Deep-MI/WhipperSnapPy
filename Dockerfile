FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libfontconfig1 \
    libdbus-1-3 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY . /WhipperSnapPy
RUN pip install /WhipperSnapPy[video]

ENTRYPOINT ["whippersnap4"]
CMD ["--help"]
