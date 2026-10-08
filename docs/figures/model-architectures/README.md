# Poster-ready model architecture figures

These figures are simplified from the implementation-level diagrams in
[`docs/model_architectures.md`](../../model_architectures.md). They use a shared
high-contrast palette, large labels, and restrained detail for viewing on a
conference poster or presentation screen.

## Recommended files

| Figure | Use |
| --- | --- |
| `00-family-overview` | One-panel overview of every model family |
| `01-v5-unet-gan` | V5 / V5.1 generator and adversarial discriminator |
| `02-v6-residual` | V6 residual refiner |
| `03-configurable-residual` | V6 research ablation family |
| `04-srcnn` | Chroma-adapted SRCNN baseline |
| `05-naf-style` | NAF-style refiner and internal block |
| `06-v7-polar` | V7 polar mean, uncertainty, and safe paths |
| `07-prism-family` | All Prism backbone/head choices and GAN branch |
| `08-li2026-hybrid` | Analytic–neural hybrid |
| `09-tabpfn-pilot` | Per-image TabPFN reconstruction workflow |

Use SVG in PowerPoint, Keynote, Illustrator, or Inkscape. Use PDF in LaTeX or
other print workflows. The PNG files are 2400 pixels wide for software that
cannot import vector graphics.

## Rebuild

From the repository root:

```bash
bash scripts/build_architecture_diagrams.sh
```

The script uses Mermaid CLI 11.12.0 through `npx` and an installed Chrome or
Chromium executable. Edit the `.mmd` sources, shared Mermaid configuration, or
CSS in this directory and rerun the script. LaTeX is not required to reproduce
the SVG, PDF, or PNG outputs.
