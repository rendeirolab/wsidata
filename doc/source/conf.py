from datetime import datetime

from sphinxawesome_theme.postprocess import Icons

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
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_copybutton",
    "myst_nb",
]
autoclass_content = "class"
autodoc_docstring_signature = True
autodoc_default_options = {"members": None, "undoc-members": None}
autodoc_typehints = "none"
autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_permalinks_icon = Icons.permalinks_icon
html_theme_options = {
    "github_url": "https://github.com/rendeiro/wsidata",
    "navigation_with_keys": True,
}

html_sidebars = {"installation": [], "intro": []}

jupyter_execute_notebooks = "off"

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5," r"8}: "
copybutton_prompt_is_regexp = True
