FROM ubuntu:20.04

# Install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip xvfb libglib2.0-0 && \
  apt clean && \
  rm -rf /var/libs/apt/lists/* /tmp/* /var/tmp/*

# Install python packages
RUN pip3 install pyopengl glfw pillow numpy pyrr PyQt5==5.15.6

COPY . /WhipperSnapPy
RUN pip3 install /WhipperSnapPy

ENTRYPOINT ["xvfb-run","whippersnap"]
CMD ["--help"]
