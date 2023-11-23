# Physical Informed Neural Networks (PINNs) for Solving Partial Differential Equations (PDEs)

This repository contains example for PINN with curriculum learning for solving PDEs. The code is based on the paper [Krishnapriyan, Aditi, et al. "Characterizing possible failure modes in physics-informed neural networks." Advances in Neural Information Processing Systems 34 (2021): 26548-26560.](https://proceedings.neurips.cc/paper/2021/file/df438e5206f31600e6ae4af72f2725f1-Paper.pdf) ([Archive Link](https://arxiv.org/abs/2109.01050))

## Overview

# To Do

## Implementation

- [ ] Technological stack (e.g. Weights & Biases, PyTorch Lightning, etc.)?
- [ ] Project structure (e.g. `src`, `data`, `notebooks`, `reports`, `references`, `tests`, `docs`, etc. In `src`: `models`, `data`, `utils`, etc., But highly depends on the project goals)
- [ ] Experimental setup (e.g. similar to the paper?)
- [ ] Implementation of curriculum learning (also in comparison to non-curriculum learning)
- [ ] Visualization of results (e.g. plots of metrics, predictions, etc.)
- [ ] Documentation (e.g. `README.md`, `requirements.txt`, `setup.py`, `LICENSE`, Code Documentation etc.)

## Other

- [ ] Project description (e.g. what is the goal of the project? what Milestones are there? What metrics should be used?)

- [ ] Question regarding paper
  - [ ] Equation (4) in paper: If $L_{u_b}$ and $L_{u_0}$ measure initial and boundary conditions, do we not miss some other NN measures, as we only rely on the PDE loss? (e.g. MSE for some u(x) or something like that?)
  - [ ] What exactly do they mean with sharp features ()
  - [ ] Figure 1: Is it fitting the boundary conditions? 
  - [ ] Their description of the convection problem is actually the convection-diffusion problem without diffusion and a constant velocity field. Hence $\beta$ describes the velocity in this case and the higher $\beta$ the higher the velocity. Is this correct?
  - [ ] The Loss described in equation (7) for the convection problem is MSE + reg. PDE term + Boundary term. What exactly do they mean with $u^i_0$ Is it the initial loss as in (4)?


- [ ] Report
  - [ ] Structure, content, etc. 