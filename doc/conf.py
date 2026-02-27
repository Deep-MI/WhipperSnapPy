# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import inspect
from datetime import date
from importlib import import_module
from typing import Dict, Optional

import whippersnappy

project = "WhipperSnapPy"
author = "Martin Reuter"
copyright = f"{date.today().year}, {author}"
release = whippersnappy.__version__
package = whippersnappy.__name__
gh_url = "https://github.com/Deep-MI/WhipperSnapPy"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "5.0"

# The document name of the “root” document, that is, the document that contains
# the root toctree directive.
root_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "numpydoc",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_design",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
]

# .md files are included via '.. include:: :parser: myst_parser.sphinx_'
# in the RST stubs; they must NOT be registered as standalone Sphinx source
# documents or autosectionlabel will produce duplicate-label warnings from
# both the real file and the doc/ symlink.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "README.md",      # symlinked from root — included inline via rst, not as a page
    "../*.md",        # exclude root-level .md files
]

templates_path = ["_templates"]

# Sphinx will warn about all references where the target cannot be found.
nitpicky = False
nitpick_ignore = []

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = [f"{package}."]

# The name of a reST role (builtin or Sphinx extension) to use as the default
# role, that is, for text marked up `like this`. This can be set to 'py:obj' to
# make `filter` a cross-reference to the Python function “filter”.
default_role = "py:obj"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "furo"
html_static_path = ["_static"]
html_title = project
html_show_sphinx = False

# Documentation to change footer icons:
# https://pradyunsg.me/furo/customisation/footer/#changing-footer-icons
html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": gh_url,
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# -- autosummary -------------------------------------------------------------
# API stubs use automodule directly — no generated/ dir needed.
autosummary_generate = False

# -- autodoc -----------------------------------------------------------------
autodoc_typehints = "none"
autodoc_member_order = "groupwise"
autodoc_warningiserror = True
autoclass_content = "class"

# Mock modules that may not be available in the doc builder environment
# (PyQt6, OpenGL, GLFW, pythreejs, etc.). Adjust this list if your builder
# provides any of these packages.
autodoc_mock_imports = [
    "PyQt6",
    "PyQt6.QtWidgets",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "glfw",
    "OpenGL",
    "OpenGL.GL",
    "OpenGL.GL.shaders",
    "pythreejs",
    "ipywidgets",
    "pyopengl",
    "pyrr",
    "PIL",
    "matplotlib",
]

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable", None),
    "mne": ("https://mne.tools/stable/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
intersphinx_timeout = 5

# -- sphinx-issues -----------------------------------------------------------
issues_github_path = gh_url.split("https://github.com/")[-1]

# -- autosectionlabels -------------------------------------------------------
autosectionlabel_prefix_document = True

# -- numpydoc ----------------------------------------------------------------
# Whether to create a Sphinx table of contents for the lists of class methods and attributes.
# If a table of contents is made, Sphinx expects each entry to have a separate page. True by default.
#numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = False  # dataclass fields → Attributes, not Parameters
numpydoc_show_class_members = False

# x-ref
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    # Matplotlib
    "Axes": "matplotlib.axes.Axes",
    "Figure": "matplotlib.figure.Figure",
    # Python
    "bool": ":class:`python:bool`",
    "Path": "pathlib.Path",
    "TextIO": "io.TextIOBase",
    # Scipy
    "csc_matrix": "scipy.sparse.csc_matrix",
}
#numpydoc_xref_ignore = {}

# validation
# https://numpydoc.readthedocs.io/en/latest/validation.html#validation-checks
error_ignores = {
    "GL01",  # docstring should start in the line immediately after the quotes
    "EX01",  # section 'Examples' not found
    "ES01",  # no extended summary found
    "SA01",  # section 'See Also' not found
    "RT02",  # The first line of the Returns section should contain only the type, unless multiple values are being returned  # noqa
}
numpydoc_validate = True
numpydoc_validation_checks = {"all"} | set(error_ignores)
numpydoc_validation_exclude = {  # regex to ignore during docstring check
    r"\.__getitem__",
    r"\.__contains__",
    r"\.__hash__",
    r"\.__mul__",
    r"\.__sub__",
    r"\.__add__",
    r"\.__iter__",
    r"\.__div__",
    r"\.__neg__",
    # Imported third-party objects exposed in plot3d module
    r"\.HTML$",
    r"\.VBox$",
    # stdlib dataclasses re-exported into cli module scope
    r"\.dataclass$",
    r"\.field$",
}

# -- sphinxcontrib-bibtex ----------------------------------------------------
bibtex_bibfiles = ["./references.bib"]

# -- sphinx.ext.linkcode -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html


def linkcode_resolve(domain: str, info: Dict[str, str]) -> Optional[str]:
    """Determine the URL corresponding to a Python object.

    Parameters
    ----------
    domain : str
        One of 'py', 'c', 'cpp', 'javascript'.
    info : dict
        With keys "module" and "fullname".

    Returns
    -------
    url : str | None
        The code URL. If None, no link is added.
    """
    if domain != "py":
        return None  # only document python objects

    # retrieve pyobject and file
    try:
        module = import_module(info["module"])
        pyobject = module
        for elt in info["fullname"].split("."):
            pyobject = getattr(pyobject, elt)
        fname = inspect.getsourcefile(pyobject).replace("\\", "/")
    except Exception:
        # Either the object could not be loaded or the file was not found.
        # For instance, properties will raise.
        return None

    # retrieve start/stop lines
    source, start_line = inspect.getsourcelines(pyobject)
    lines = "L%d-L%d" % (start_line, start_line + len(source) - 1)

    # create URL
    if "dev" in release:
        branch = "main"
    else:
        return None  # alternatively, link to a maint/version branch
    fname = fname.rsplit(f"/{package}/")[1]
    url = f"{gh_url}/blob/{branch}/{package}/{fname}#{lines}"
    return url


# # -- sphinx-gallery ----------------------------------------------------------
# sphinx_gallery_conf = {
#     "backreferences_dir": "generated/backreferences",
#     "doc_module": (f"{package}",),
#     "examples_dirs": ["generated/examples"],
#     "exclude_implicit_doc": {},  # set
#     "filename_pattern": r"\d{2}_",
#     "gallery_dirs": ["generated/examples"],
#     "line_numbers": False,
#     "plot_gallery": True,
#     "reference_url": {f"{package}": None},
#     "remove_config_comments": True,
#     "show_memory": True,
#     "within_subsection_order": FileNameSortKey,
# }


# -- nbsphinx ----------------------------------------------------------------
# Re-execute notebooks during the Sphinx build so outputs appear in the docs.
# Notebooks are executed with the kernel specified by nbsphinx_kernel_name.
# The sample data is fetched from the GitHub release assets (or from the
# local sub-rs/ directory in the repo when the release is not yet published).
nbsphinx_execute = "auto"

# Kernel to use for execution (must be installed: pip install ipykernel).
nbsphinx_kernel_name = "python3"

# Maximum execution time per cell (seconds).
nbsphinx_timeout = 600

# -- make sure pandoc gets installed -----------------------------------------
from inspect import getsourcefile
import os

# Get path to directory containing this file, conf.py.
DOCS_DIRECTORY = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))

def ensure_pandoc_installed(_):
    try:
        import pypandoc
        pandoc_dir = os.path.join(DOCS_DIRECTORY, "bin")
        if pandoc_dir not in os.environ["PATH"].split(os.pathsep):
            os.environ["PATH"] += os.pathsep + pandoc_dir
        pypandoc.ensure_pandoc_installed(
            targetfolder=pandoc_dir,
            delete_installer=True,
        )
    except Exception:
        pass  # pandoc already on PATH (CI) or download failed (local SSL) — continue


def setup(app):
    app.connect("builder-inited", ensure_pandoc_installed)
    # nbsphinx omits image/gif from its MIME-type pipeline, so animated GIFs
    # produced by display(IPython.display.Image(...)) fall back to the
    # text/plain repr.  Three patches are needed:
    #
    #  1. DISPLAY_DATA_PRIORITY_HTML — tell nbsphinx to *select* image/gif output.
    #  2. RST_TEMPLATE — tell the Jinja template to render image/gif via the
    #     standard ``.. image::`` directive (same as image/png / image/jpeg).
    #  3. ExtractOutputPreprocessor.extract_output_types — tell nbconvert to
    #     actually extract the GIF bytes to a file so the directive has a path.
    import logging
    _logger = logging.getLogger(__name__)
    try:
        import nbsphinx as _nbsphinx

        # 1. Priority list
        if "image/gif" not in _nbsphinx.DISPLAY_DATA_PRIORITY_HTML:
            _priority = list(_nbsphinx.DISPLAY_DATA_PRIORITY_HTML)
            try:
                idx = _priority.index("image/jpeg")
                _priority.insert(idx + 1, "image/gif")
            except ValueError:
                import warnings
                warnings.warn(
                    "whippersnappy/conf.py: 'image/jpeg' not found in "
                    "nbsphinx.DISPLAY_DATA_PRIORITY_HTML; appending "
                    "'image/gif' at the end instead. The GIF may still "
                    "render correctly, but priority ordering is unknown.",
                    stacklevel=2,
                )
                _priority.append("image/gif")
            _nbsphinx.DISPLAY_DATA_PRIORITY_HTML = tuple(_priority)

        # 2. RST template — add image/gif alongside the other raster types.
        #    We verify the substitution actually changed something; if the
        #    upstream template text has changed, warn so the breakage is visible
        #    rather than silently producing a broken GIF rendering.
        import warnings as _warnings
        _RST_OLD = "datatype in ['image/svg+xml', 'image/png', 'image/jpeg', 'application/pdf']"
        _RST_NEW = "datatype in ['image/svg+xml', 'image/png', 'image/jpeg', 'image/gif', 'application/pdf']"
        _patched_template = _nbsphinx.RST_TEMPLATE.replace(_RST_OLD, _RST_NEW)
        if _patched_template == _nbsphinx.RST_TEMPLATE:
            _warnings.warn(
                "whippersnappy/conf.py: could not patch nbsphinx.RST_TEMPLATE "
                "to add 'image/gif' support — the expected substring was not "
                "found. Animated GIFs may not render in the documentation. "
                "The nbsphinx template may have changed upstream; please update "
                "the patch in doc/conf.py.",
                stacklevel=2,
            )
        else:
            _nbsphinx.RST_TEMPLATE = _patched_template

        # 3. nbconvert extractor — ExtractOutputPreprocessor hard-codes
        #    {"image/png", "image/jpeg", "application/pdf"} as the types that
        #    get base64-decoded to binary.  image/gif falls into the "else: text"
        #    branch and is written as a raw base64 string, producing a corrupt
        #    file.  Two sub-patches fix this:
        #      3a  add "image/gif" to extract_output_types so the extractor
        #          visits it at all.
        #      3b  wrap preprocess_cell to strip gif data before the parent runs,
        #          then decode it to bytes and inject the result into resources.
        from nbconvert.preprocessors import ExtractOutputPreprocessor as _EOP
        from binascii import a2b_base64 as _a2b

        # 3a — register image/gif in extract_output_types via __init__ patch
        _eop_orig_init = _EOP.__init__
        def _eop_patched_init(self, *args, **kwargs):
            _eop_orig_init(self, *args, **kwargs)
            self.extract_output_types = self.extract_output_types | {"image/gif"}
        _EOP.__init__ = _eop_patched_init

        # 3b — patch preprocess_cell to handle image/gif as binary (base64 → bytes).
        #      The parent hard-codes only png/jpeg/pdf for binary decode; gif falls
        #      into the "else: text" branch and would be written as a raw base64
        #      string (corrupt file).  We strip gif data before calling the parent,
        #      decode it ourselves, and store the binary bytes in resources.
        _eop_orig_preprocess_cell = _EOP.preprocess_cell

        def _eop_patched_preprocess_cell(self, cell, resources, cell_index):
            # Before calling the original, convert any image/gif from base64
            # string to bytes — but the original then hits the
            # `not isinstance(data, str)` → json branch for bytes, so we must
            # pre-decode AND bypass the parent entirely for image/gif outputs.
            #
            # Strategy: strip image/gif from outputs before calling parent,
            # then handle extraction ourselves, then restore.
            import os as _os
            gif_extractions = []  # list of (out, raw_b64) to process after parent

            for out in cell.get("outputs", []):
                if out.get("output_type") not in ("display_data", "execute_result"):
                    continue
                data = out.get("data", {})
                if "image/gif" in data and isinstance(data["image/gif"], str):
                    gif_extractions.append((out, data.pop("image/gif")))

            # Run original preprocessor (without image/gif in data)
            cell, resources = _eop_orig_preprocess_cell(self, cell, resources, cell_index)

            if not gif_extractions:
                return cell, resources

            # Now handle image/gif extractions ourselves
            unique_key = resources.get("unique_key", "output")
            output_files_dir = resources.get("output_files_dir", None)
            if not isinstance(resources.get("outputs"), dict):
                resources["outputs"] = {}

            outputs_list = cell.get("outputs", [])
            for out, raw_b64 in gif_extractions:
                # Restore the b64 string in the cell data for the RST template
                out["data"]["image/gif"] = raw_b64
                # Find the index of this output in the cell
                try:
                    index = outputs_list.index(out)
                except ValueError:
                    index = 0
                # Build filename
                filename = self.output_filename_template.format(
                    unique_key=unique_key,
                    cell_index=cell_index,
                    index=index,
                    extension=".gif",
                )
                if output_files_dir is not None:
                    filename = _os.path.join(output_files_dir, filename)
                # Store binary GIF bytes in resources
                resources["outputs"][filename] = _a2b(raw_b64)
                # Store filename in output metadata so the Jinja template uses it
                if "metadata" not in out:
                    out["metadata"] = {}
                if "filenames" not in out["metadata"]:
                    out["metadata"]["filenames"] = {}
                out["metadata"]["filenames"]["image/gif"] = filename

            return cell, resources

        _EOP.preprocess_cell = _eop_patched_preprocess_cell
    except ImportError as exc:
        _logger.warning(
            "conf.py: could not patch nbsphinx/nbconvert for GIF support "
            "(package not installed): %s. Animated GIFs will not render.",
            exc,
        )
    except AttributeError as exc:
        _logger.warning(
            "conf.py: nbsphinx or nbconvert API has changed and the GIF patch "
            "could not be applied: %s. Animated GIFs will not render. "
            "Please update the patch in doc/conf.py.",
            exc,
        )
