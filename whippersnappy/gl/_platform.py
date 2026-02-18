"""Bootstrap PyOpenGL platform selection — must be imported first.

Imported unconditionally at the top of gl/__init__.py before any other
OpenGL symbol. Sets PYOPENGL_PLATFORM=egl when no display is available
so that PyOpenGL uses the EGL backend instead of GLX.

If the user has already set PYOPENGL_PLATFORM, that value is respected.
"""
import os

if "PYOPENGL_PLATFORM" not in os.environ:
    # No display = definitely headless, force EGL now before any import
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ["PYOPENGL_PLATFORM"] = "egl"
