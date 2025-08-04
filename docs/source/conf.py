from datetime import datetime
from functools import cached_property

import wsidata

project = "wsidata"
copyright = f"{datetime.now().year}, Rendeiro Lab"
author = "Yimin Zheng"
release = wsidata.__version__

extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_copybutton",
    "myst_nb",
    "sphinxext.opengraph",
]
autoclass_content = "class"
autodoc_docstring_signature = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "no-undoc-members": True,
    "special-members": "__call__",
    "exclude-members": "__init__, __weakref__",
    "class-doc-from": "class",
}
autodoc_typehints = "none"
autosummary_generate = True
numpydoc_show_class_members = False
add_module_names = False

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/logo@3x.png"
html_css_files = ["custom.css"]
html_theme_options = {
    "github_url": "https://github.com/rendeirolab/wsidata",
    "navigation_with_keys": True,
    "show_prev_next": False,
}

html_sidebars = {"installation": [], "intro": []}

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5," r"8}: "
copybutton_prompt_is_regexp = True

intersphinx_mapping = {
    "spatialdata": ("https://spatialdata.scverse.org/en/latest", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "lazyslide": ("https://lazyslide.readthedocs.io/en/latest/", None),
    "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/latest", None),
}
