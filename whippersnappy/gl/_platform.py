"""Bootstrap PyOpenGL platform selection — must be imported first.

Imported unconditionally at the top of gl/__init__.py before any other
OpenGL symbol. Sets PYOPENGL_PLATFORM=egl when running headless on Linux
so that PyOpenGL uses the EGL backend instead of GLX.

On macOS PyOpenGL uses CGL and on Windows it uses WGL — both are handled
natively without EGL. If the user has already set PYOPENGL_PLATFORM that
value is always respected.
"""
import os
import sys

if "PYOPENGL_PLATFORM" not in os.environ and sys.platform == "linux":
    # No X11/Wayland display on Linux → force EGL headless backend.
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ["PYOPENGL_PLATFORM"] = "egl"
