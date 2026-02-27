FROM python:3.11-slim

# libosmesa6  — OSMesa software renderer for headless OpenGL (no GPU/display needed)
# libgl1       — base OpenGL shared library required by PyOpenGL
# libglib2.0-0, libfontconfig1, libdbus-1-3 — runtime deps for Pillow / font rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libosmesa6 \
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
