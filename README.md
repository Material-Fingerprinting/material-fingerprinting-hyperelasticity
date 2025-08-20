![Material Fingerprinting](plots/logo.png)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16778796.svg)](https://doi.org/10.5281/zenodo.16778796)

We propose [Material Fingerprinting](https://doi.org/10.48550/arXiv.2508.07831), a new method for the rapid discovery of mechanical material models from direct or indirect data that avoids solving potentially non-convex optimization problems. The core assumption of Material Fingerprinting is that each material exhibits a unique response when subjected to a standardized experimental setup. This response can be interpreted as the material's fingerprint, essentially a unique identifier that encodes all pertinent information about the material's mechanical characteristics. Consequently, if a database containing fingerprints and their corresponding mechanical models is established during an offline phase, an unseen material can be characterized rapidly in an online phase. This is accomplished by measuring its fingerprints and employing a pattern recognition algorithm to discover the best matching fingerprint in the database.

![Material Fingerprinting](plots/abstract.png)

The figure above illustrates the concept of Material Fingerprinting in both direct and indirect experimental setups. The supervised case involves homogeneous deformation fields, yielding direct strain-stress data pairs. The unsupervised case, in contrast, uses complex specimen geometries that produce heterogeneous deformation fields and only provide indirect displacement and force measurements.

![Material Fingerprinting](plots/pattern_recognition_matrices.png)

At the core of Material Fingerprinting is a straightforward pattern recognition algorithm. The figure above demonstrates how a new measurement is compared against all fingerprints in the database, correctly discovering the underlying material model — in this case, the Ogden model.

## About this repository

This repository provides the code and data accompanying the publication [Material Fingerprinting: A shortcut to material model discovery without solving optimization problems](https://doi.org/10.48550/arXiv.2508.07831).
To ensure reproducibility, the contents of this repository remain unchanged.
For the actively maintained, pip-installable version of Material Fingerprinting, please visit [https://github.com/Material-Fingerprinting/material-fingerprinting](https://github.com/Material-Fingerprinting/material-fingerprinting).

## Reference

```bibtex
@article{flaschel2025material,
  author       = {Flaschel, Moritz and Martonová, Denisa and Veil, Carina and Kuhl, Ellen},
  title        = {Material Fingerprinting: A shortcut to material model discovery without solving optimization problems},
  year         = {2025},
  doi          = {10.48550/arXiv.2508.07831},
}
```

## How to cite the code

```bibtex
@software{flaschel2025materialfingerprintinghyperelasticity,
  author       = {Flaschel, Moritz and Martonová, Denisa and Veil, Carina and Kuhl, Ellen},
  title        = {Supplementary software for "Material Fingerprinting: A shortcut to material model discovery without solving optimization problems"},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.16778796},
  url          = {https://github.com/Material-Fingerprinting/material-fingerprinting-hyperelasticity}
}
```
