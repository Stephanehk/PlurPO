# PlurPO project page

Static site for *Mitigating social sycophancy via Pluralistic Preference Optimization*. No build step.

- `index.html`: page content
- `style.css`: styles (light and dark mode)
- `app.js`: stakeholder veto demo, results dot plot, examples. Results numbers are in the `MODELS` array.
- `images/`: AITA and AITA-Flipped figures from the paper

## Preview locally

```
python3 -m http.server 8000
```

Then open http://localhost:8000.

## Deployment

Served by GitHub Pages from the `gh-pages` branch of https://github.com/Stephanehk/PlurPO at https://stephanehk.github.io/PlurPO/. Push changes to `gh-pages` to update the site.

## Before publishing

Replace the `#` link for Paper in `index.html` and update the BibTeX entry with the arXiv ID once the paper is posted.
