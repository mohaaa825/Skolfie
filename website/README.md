# Skolfie website

A static landing page for teachers and schools (plain HTML, CSS and JS; no build step).
Palette: white, baby blue and pink, matching `THEME_TOKENS` in `skolfie/app/gradio_app.py`.

- `index.html` page, `styles.css` look, `main.js` menu/tabs/links, `config.js` settings.
- **Link the checker:** set `appUrl` in `config.js` to the app link (for example the Gradio
  share link from `notebooks/app.ipynb`, or your own server). Until it is set, the checker buttons
  read "Request access"; once it is set they switch to "Open the checker" / "Check a text" (their
  `data-live-label`) and open the app.
- **Contact:** set `contactEmail` in `config.js`. Then "Request access" opens an email to that
  address; without it, those buttons scroll to the "Try Skolfie" section, which has no way to reach you.
- **Font:** Neue Montreal, like vesence.com. It's a commercial font from Pangram Pangram
  (free for personal use only), so it isn't bundled; the site falls back to Hanken Grotesk, a
  free look-alike. After buying a web licence, put the files in `assets/fonts/` and add an
  `@font-face { font-family: "Neue Montreal"; src: url(assets/fonts/NeueMontreal-Regular.woff2) format("woff2"); font-weight: 400; }`
  (plus one for Medium, weight 500) at the top of `styles.css`.
- **Landing page background:** no photo, on purpose. A pale sky tint with faint ruled lines (like
  lined paper), drawn in CSS (`.hero__bg`), so the product example is the first thing people see.
  The earlier cherry-blossom painting (`hero-blossom.jpg`) is in the git history if it is ever needed.
- **Background photos:** `try-harbour.jpg` (behind the "Try Skolfie" glass card), with a `-sm`
  version for phones, is a painting supplied by the site owner. `aerial-*.jpg` are three crops of an aerial Stockholm painting (also supplied) on the who-it's-for cards. Keep each file under 400 KB.
- **Motion:** kept quiet on purpose, since this is a tool schools need to trust: short scroll
  fade-ins, and the mock-up's meter and highlights animating in (they show what the product does).
  No hover lifts, button shine, drifting or parallax photos, or counting numbers. All motion
  switches off for visitors who ask their device for reduced motion.
- **Preview locally:** `python -m http.server -d website 8000`, then open http://localhost:8000.
- **Publish:** GitHub Pages via `.github/workflows/pages.yml`. Once: Settings -> Pages ->
  Source "GitHub Actions". Each push to `main` that touches `website/` redeploys.

The numbers on the page come from the 8 Nov 2026 models (`My Drive/skolfie/reports_modern/`);
update them after a retrain.
