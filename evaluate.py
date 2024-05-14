from src.runner import GraphRunner, MLPRunner
from sys import argv
import numpy as np

import yaml
import os.path as osp
    
if __name__ == "__main__":
    if len(argv) < 2:
        print("Run the program with '$ python evaluate.py <run_name>'")
        exit(1)
    
    run_name = argv[1]
    with open(osp.join("runs", run_name, "config.yaml"), 'r') as file:
        config = yaml.safe_load(file)
    
    steps = config["steps"]
    
    trials = 100
    total_rewards = []
    episodes_collisions = []
    episodes_finished = []
    episodes_duration = []

    for i in range(trials):
        
        if config["type"] == 'graph':
            runner = GraphRunner(config=config)
        else:
            runner = MLPRunner(config=config)

        print(f'{str(i+1)}/{str(trials)}', end='')
        
        total_reward, episode_collisions, episode_finished, episode_duration = runner.eval(
            num_steps=steps, 
            run_name=run_name, 
            # render=True, 
            load=True
        )
        
        total_rewards.append(total_reward)
        episodes_collisions.append(episode_collisions)
        episodes_finished.append(episode_finished)
        episodes_duration.append(episode_duration)

        
    print(f'Reward = {round(np.mean(total_rewards), 2)}') 
    print(f'T = {round(np.mean(episodes_duration), 2)}') 
    print(f'# col = {round(np.mean(episodes_collisions), 2)}') 
    print(f'S% = {round(np.mean(episodes_finished)*100, 2)}') 
    