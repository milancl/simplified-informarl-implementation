## [Re] Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation

<b>Jacopo di Ventura, Matteo Martinoli, Claudio Milanesi</b>

This repo contains our simplified implementation for InforMARL. 

## Repo Structure 

The scripts to start training, evaluation and multiple training can be found in the root folder as `training.py`, `evaluate.py`, and `multiple_training.py`. The file `analyze.py` contains basic code for creating plots given a run.
- To start the training, modify the parameters in `train.py` and run `$python training.py [run_name]`. If a parameter `run_name` is given, that run should be present in the `runs/` folder as `runs/run_name` and the training will start with the models checkpoints and with the same `config.yaml`
- To perform evaluation of a given run, run `$python evaluate.py <run_name>`.
- `multiple_training.py` is a script used to launch the graph and mlp training for 3, 7, and 10 agents.

## Installation
For simplicity, we saved our environment installed packages in `requirements.txt` file. This project was created using `Python 3.12.2`. To install all necessary packages, run
`$pip install -r requirements.txt` and then `$pip install PyYAML`.
