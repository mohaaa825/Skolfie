# Skolfie website

A static landing page for teachers and schools (plain HTML, CSS and JS; no build step).
Palette: white, baby blue and pink, matching `THEME_TOKENS` in `skolfie/app/gradio_app.py`.

- `index.html` page, `styles.css` look, `main.js` menu/tabs/links, `config.js` settings.
- **Link the checker:** set `appUrl` in `config.js` to the app link (for example the Gradio
  share link from `notebooks/app.ipynb`, or your own server). Empty = the "Open the checker"
  buttons scroll to the "Try Skolfie" section instead.
- **Contact:** set `contactEmail` in `config.js` to show "Ask about access for your school".
- **Preview locally:** `python -m http.server -d website 8000`, then open http://localhost:8000.
- **Publish:** GitHub Pages via `.github/workflows/pages.yml`. Once: Settings -> Pages ->
  Source "GitHub Actions". Each push to `main` that touches `website/` redeploys.

The numbers on the page come from the 8 Nov 2026 models (`My Drive/skolfie/reports_modern/`);
update them after a retrain.
