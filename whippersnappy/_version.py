"""Version number."""

try:
    from importlib.metadata import version
    __version__ = version(__package__)
except Exception:
    # Fallback when package is not installed (e.g., running from source)
    __version__ = "1.4.0-dev"
