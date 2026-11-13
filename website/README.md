# Skolfie website

A static landing page for teachers and schools (plain HTML, CSS and JS; no build step).
Palette: white, baby blue and pink, matching `THEME_TOKENS` in `skolfie/app/gradio_app.py`.

- `index.html` page, `styles.css` look, `main.js` menu/tabs/links, `config.js` settings.
- **Link the checker:** set `appUrl` in `config.js` to the app link (for example the Gradio
  share link from `notebooks/app.ipynb`, or your own server). Empty = the "Open the checker"
  buttons scroll to the "Try Skolfie" section instead.
- **Contact:** set `contactEmail` in `config.js` to show "Ask about access for your school".
- **Font:** Neue Montreal, like vesence.com. It's a commercial font from Pangram Pangram
  (free for personal use only), so it isn't bundled; the site falls back to Hanken Grotesk, a
  free look-alike. After buying a web licence, put the files in `assets/fonts/` and add an
  `@font-face { font-family: "Neue Montreal"; src: url(assets/fonts/NeueMontreal-Regular.woff2) format("woff2"); font-weight: 400; }`
  (plus one for Medium, weight 500) at the top of `styles.css`.
- **Background photos:** `assets/photos/hero-blossom.jpg` (landing page, Stockholm waterfront with
  cherry blossoms) and `try-harbour.jpg` (behind the "Try Skolfie" glass card), each with a `-sm`
  version for phones. Both are paintings supplied by the site owner. `aerial-*.jpg` are three crops of an aerial Stockholm painting (also supplied) on the who-it's-for cards. Keep each file under 400 KB.
- **Motion:** headline word reveal, scroll fade-ups, count-up numbers, the mock-up's meter and
  highlights animating in, slow drift and parallax on the photos. All of it switches off for
  visitors who ask their device for reduced motion.
- **Preview locally:** `python -m http.server -d website 8000`, then open http://localhost:8000.
- **Publish:** GitHub Pages via `.github/workflows/pages.yml`. Once: Settings -> Pages ->
  Source "GitHub Actions". Each push to `main` that touches `website/` redeploys.

The numbers on the page come from the 8 Nov 2026 models (`My Drive/skolfie/reports_modern/`);
update them after a retrain.
