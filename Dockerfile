FROM python:3.11-slim

# libosmesa6  — OSMesa CPU software renderer (default headless path, no GPU needed)
# libegl1      — EGL dispatch library; enables GPU rendering when /dev/dri/renderD*
#                is passed via --device (e.g. docker run --device /dev/dri/renderD128)
# libgl1       — base OpenGL dispatch library required by PyOpenGL
# libglib2.0-0, libfontconfig1, libdbus-1-3 — runtime deps for Pillow / font rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libosmesa6 \
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
