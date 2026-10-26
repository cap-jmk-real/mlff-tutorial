---

# 🧠⚛️ `mlff-tutorial`

Tutorials on building Machine Learning Force Fields (MLFFs) and running Molecular Dynamics (MD) simulations for complex molecular systems using **PyTorch**, **OpenMM**, **ASE**, and **mond**.

This project serves as a hands-on guide for developing and applying ML-based potentials in atomistic simulations, bridging the gap between modern deep learning and computational chemistry.

---

## 🚀 What You'll Learn

* Constructing ML-based force fields using PyTorch & PyTorch Geometric
* Setting up atomistic datasets with ASE and mond
* Running GPU-accelerated MD simulations in OpenMM with custom force fields
* Evaluating and visualizing simulation results
* Handling real molecular data and integrating DFT reference calculations

---

## 🛠 Technologies Used

* [PyTorch](https://pytorch.org/)
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
* [OpenMM](https://openmm.org/)
* [ASE (Atomic Simulation Environment)](https://wiki.fysik.dtu.dk/ase/)
* [mond](https://github.com/cap-jmk-real/mond) (custom data preparation and featurization tools)

---


## 🔧 Getting Started

1. **Clone the repository**

```bash
git clone https://github.com/cap-jmk-real/mlff-tutorial.git
cd mlff-tutorial
```

2. **Install dependencies**

It is recommmended to use `conda` to manage environments. Each tutorial introduces you how to best setup an environment for that tutorial. It can't be guaranteed that all environments work for all tutorials.

For AIMD tutorials (notebook 02+), install the **vendored `mond` package from this repo** so mass/force units in `velocity_verlet_step` match Psi4/ASE (eV, eV/Å):

```bash
pip install -e .
```

Unit conventions and dataset regeneration notes: `docs/UNITS.md`. 

3. **Run your first tutorial**

```bash
jupyter lab notebooks/
```


## 🤝 Collaboration

This project is part of my scientific portfolio.
Feel free to [open an issue](https://github.com/julian-kl/mlff-tutorial/issues) or contact me if you're working on similar problems or want to collaborate.

---

## 📜 License

MIT License — see `LICENSE` file for details.

---
