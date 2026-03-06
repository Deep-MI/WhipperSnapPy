FROM python:3.11-slim

# Suppress Mesa's shader-cache warning ("Failed to create //.cache …") that
# appears when running as a non-standard user inside Docker where $HOME is
# unset or points to a non-writable directory.
ENV MESA_SHADER_CACHE_DISABLE=1

# libegl1     — GLVND EGL dispatch library (routes to GPU or Mesa llvmpipe)
# libosmesa6  — OSMesa CPU fallback for environments where EGL cannot initialise
# libgl1      — base OpenGL dispatch library required by PyOpenGL
# libglib2.0-0, libfontconfig1, libdbus-1-3 — runtime deps for Pillow / font rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 \
    libosmesa6 \
    libgl1 \
    libglib2.0-0 \
    libfontconfig1 \
    libdbus-1-3 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Create the GLVND EGL vendor directory so the NVIDIA Container Runtime can
# inject its EGL ICD (10_nvidia.json) when --gpus all is passed.
# Without this directory the NVIDIA EGL ICD is silently not registered and
# EGL falls back to Mesa llvmpipe even with a GPU available.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d

RUN pip install --upgrade pip

COPY . /WhipperSnapPy
RUN pip install /WhipperSnapPy[video]

ENTRYPOINT ["whippersnap4"]
CMD ["--help"]
