"""Configuration and system-info helpers (top-level module)."""

import platform
import re
import sys
from functools import partial
from importlib.metadata import requires, version
from typing import IO, Callable, Optional

import psutil


def sys_info(fid: Optional[IO] = None, developer: bool = False):
    """Print the system information for debugging.

    Parameters
    ----------
    fid : file-like, default=None
        The file to write to, passed to :func:`print`.
        Can be None to use :data:`sys.stdout`.
    developer : bool, default=False
        If True, display information about optional dependencies.
    """
    ljust = 26
    out = partial(print, end="", file=fid)
    package = __package__.split(".")[0]

    # OS information - requires python 3.8 or above
    out("Platform:".ljust(ljust) + platform.platform() + "\n")
    # Python information
    out("Python:".ljust(ljust) + sys.version.replace("\n", " ") + "\n")
    out("Executable:".ljust(ljust) + sys.executable + "\n")
    # CPU information
    out("CPU:".ljust(ljust) + platform.processor() + "\n")
    out("Physical cores:".ljust(ljust) + str(psutil.cpu_count(False)) + "\n")
    out("Logical cores:".ljust(ljust) + str(psutil.cpu_count(True)) + "\n")
    # Memory information
    out("RAM:".ljust(ljust))
    out(f"{psutil.virtual_memory().total / float(2 ** 30):0.1f} GB\n")
    out("SWAP:".ljust(ljust))
    out(f"{psutil.swap_memory().total / float(2 ** 30):0.1f} GB\n")

    # dependencies
    out("\nDependencies info\n")
    # package version may not be available when running tests from the
    # repository root (package not installed). Handle gracefully.
    try:
        pkg_version = version(package)
    except Exception:
        pkg_version = "Not installed."
    out(f"{package}:".ljust(ljust) + pkg_version + "\n")

    try:
        raw_requires = requires(package) or []
    except Exception:
        raw_requires = []

    # If package metadata is not present (e.g. running from source), try to
    # read dependencies declared in pyproject.toml so tests running against
    # the tree without installing still report expected deps.
    if not raw_requires:
        try:
            from pathlib import Path
            try:
                import tomllib as _toml
            except Exception:
                _toml = None

            repo_root = Path(__file__).resolve().parents[1]
            pyproject_path = repo_root / "pyproject.toml"
            if _toml is not None and pyproject_path.exists():
                with pyproject_path.open("rb") as fh:
                    data = _toml.load(fh)
                proj = data.get("project", {})
                deps = proj.get("dependencies", []) or []
                # dependencies may be in the form 'pkg>=1.2' etc.
                raw_requires = deps
        except Exception:
            raw_requires = []

    dependencies = [elt.split(";")[0].rstrip() for elt in raw_requires if "extra" not in elt]
    _list_dependencies_info(out, ljust, dependencies)

    # extras
    if developer:
        keys = (
            "build",
            "doc",
            "test",
            "style",
        )
        for key in keys:
            try:
                raw_requires = requires(package) or []
            except Exception:
                raw_requires = []

            # If package metadata missing, fall back to pyproject.toml optional-dependencies
            if not raw_requires:
                try:
                    from pathlib import Path
                    try:
                        import tomllib as _toml
                    except Exception:
                        _toml = None

                    repo_root = Path(__file__).resolve().parents[1]
                    pyproject_path = repo_root / "pyproject.toml"
                    if _toml is not None and pyproject_path.exists():
                        with pyproject_path.open("rb") as fh:
                            data = _toml.load(fh)
                        proj = data.get("project", {})
                        opt = proj.get("optional-dependencies", {}) or {}
                        deps = opt.get(key, []) or []
                        raw_requires = deps
                except Exception:
                    raw_requires = []

            dependencies = [
                elt.split(";")[0].rstrip()
                for elt in raw_requires
                if f"extra == '{key}'" in elt or f"extra == \"{key}\"" in elt or True
            ]
            if len(dependencies) == 0:
                continue
            out(f"\nOptional '{key}' info\n")
            _list_dependencies_info(out, ljust, dependencies)


def _list_dependencies_info(out: Callable, ljust: int, dependencies: list[str]):
    """List dependencies names and versions.

    Parameters
    ----------
    out : Callable
        output function
    ljust : int
         length of returned string
    dependencies : List[str]
        list of dependencies

    """
    for dep in dependencies:
        # handle dependencies with version specifiers
        specifiers_pattern = r"(~=|==|!=|<=|>=|<|>|===)"
        specifiers = re.findall(specifiers_pattern, dep)
        if len(specifiers) != 0:
            dep, _ = dep.split(specifiers[0])
            while not dep[-1].isalpha():
                dep = dep[:-1]
        # handle dependencies provided with a [key], e.g. pydocstyle[toml]
        if "[" in dep:
            dep = dep.split("[")[0]
        try:
            version_ = version(dep)
        except Exception:
            version_ = "Not found."

        # handle special dependencies with backends, C dep, ..
        if dep in ("matplotlib", "seaborn") and version_ != "Not found.":
            try:
                import importlib
                plt = importlib.import_module("matplotlib.pyplot")
                backend = plt.get_backend()
            except Exception:
                backend = "Not found"

            out(f"{dep}:".ljust(ljust) + version_ + f" (backend: {backend})\n")

        else:
            out(f"{dep}:".ljust(ljust) + version_ + "\n")
