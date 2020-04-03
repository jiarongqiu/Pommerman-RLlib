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

### Running Script
! Make sure you are in the root of this project
* First set up the environment path
```bash
conda activate pommerman
export PYTHONPATH=$PYTHONPATH:.
```

* then running the script
```bash
python script/A2C/Train_curriculum.py game_training
```

* running the evaluation script to see your result
* Do not forget to modified your model's path in script
```bash
python script/A2C/Train_curriculum.py game_training
```
