# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "splaat"
copyright = "2025, Michael McAuliffe and Kaylynn Gunter"
author = "Michael McAuliffe and Kaylynn Gunter"
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
import splaat  # noqa
from splaat.utils import get_splaat_version  # noqa

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


version = ".".join(get_splaat_version().split(".", maxsplit=2)[:2])
# The full version, including alpha/beta/rc tags.
release = get_splaat_version()

extensions = [
    "myst_nb",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ("http", "https", "mailto")

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/fontawesome.min.css",
]

html_sidebars = {"**": ["search-field.html", "sidebar-nav-bs.html", "sidebar-ethical-ads.html"]}

html_theme_options = {
    "font": "Segoe UI",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/mmcauliffe/splaat",
            "icon": "fab fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/splaat/",
            "icon": "fas fa-box",
        },
    ],
    "logo": {
        "text": "splaat",
        # "image_dark": "logo-dark.svg",
    },
    # "show_nav_level": 1,
    # "navigation_depth": 4,
    # "show_toc_level": 2,
    # "collapse_navigation": True,
}
html_context = {
    "github_user": "mmcauliffe",
    "github_repo": "splaat",
    "github_version": "main",
    "doc_path": "docs/source",
}
