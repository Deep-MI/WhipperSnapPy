FROM python:3.11-slim

# Suppress Mesa's shader-cache warning ("Failed to create //.cache …") that
# appears when running as a non-standard user inside Docker where $HOME is
# unset or points to a non-writable directory.
ENV MESA_SHADER_CACHE_DISABLE=1

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

# Create a 'render' group (GID 103, matching Debian/Ubuntu default) so that
# GPU EGL rendering works when the host render device is passed in via
#   docker run --device /dev/dri/renderD128 --group-add render ...
# If the host render group has a different GID use --group-add <GID> instead.
RUN groupadd -g 103 render 2>/dev/null || true

RUN pip install --upgrade pip

COPY . /WhipperSnapPy
RUN pip install /WhipperSnapPy[video]

ENTRYPOINT ["whippersnap4"]
CMD ["--help"]
