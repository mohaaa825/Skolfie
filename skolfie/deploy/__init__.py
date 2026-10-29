"""
skolfie/deploy
---------------
Putting the Skolfie web app online. `hf_space` builds and publishes a
Hugging Face Space (`skolfie space build` / `skolfie space publish`, or
section (d) of notebooks/app.ipynb).
"""

from skolfie.deploy.hf_space import (
    build_space,
    collect_versions,
    publish_space,
    space_page_url,
    space_site_url,
)

__all__ = ["build_space", "collect_versions", "publish_space", "space_page_url",
           "space_site_url"]
