FROM ubuntu:20.04

# Install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 pip xvfb libglib2.0-0 libxkbcommon-x11-0 libgl1 libegl1 \
    libfontconfig1 libdbus-1-3 && \
  apt clean && \
  rm -rf /var/libs/apt/lists/* /tmp/* /var/tmp/*

# Install python packages
RUN pip install --upgrade pip
RUN pip install pyopengl glfw pillow numpy pyrr PyQt6

COPY . /WhipperSnapPy
RUN pip install /WhipperSnapPy

ENTRYPOINT ["xvfb-run","whippersnap"]
CMD ["--help"]
