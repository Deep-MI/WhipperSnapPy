FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 pip \
    libegl1 \
    libglib2.0-0 libfontconfig1 libdbus-1-3 && \
  apt clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install --upgrade pip
RUN pip install pyopengl glfw pillow numpy pyrr

COPY . /WhipperSnapPy
RUN pip install /WhipperSnapPy

ENTRYPOINT ["whippersnap"]
CMD ["--help"]
