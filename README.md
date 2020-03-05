# Pommerman-RLlib

### Environment Setup
* pommerman: download the playground and install the lib

```bash
cd playground
conda env create -f env.yml
```

To update pommerman environment
```bash
pip install .
```
* RLlib & libs:

```bash
cd Pommerman-RLlib
pip install -r requirements.txt
```

### Structure
* models: build customize PyTorch models
* demo: deploy the trained models for visualization
* env: a pommerman wrapper to incorporate with RLlib
* reward: an interface to build custom reward functions