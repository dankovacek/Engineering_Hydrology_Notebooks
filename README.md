## Jupyter Book and Binder

Launch the main notebook using Binder:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dankovacek/run_of_river_intro.git/main)


Info for [building books and hosting on Github Pages](https://jupyterbook.org/publish/gh-pages.html)

After updating any content, rebuilt the repo:

`jupyter-book build content/`

Then, update the github pages site. Use the gh-pages branch update tool:

`ghp-import -n -p -f content/_build/html`

[Visit the site](https://dankovacek.github.io/Engineering_Hydrology_Notebooks/) at Github sites

`https://dankovacek.github.io/Engineering_Hydrology_Notebooks/`

