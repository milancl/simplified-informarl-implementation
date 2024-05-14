from src.runner import GraphRunner
from sys import argv

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
    
    trials = 10
    for _ in range(trials):
        
        runner = GraphRunner(
            config=config,
        )
        
        total_rewards, episode_collisions, episode_dones, when_dones = runner.eval(
            num_steps=steps, 
            run_name=run_name, 
            # render=True, 
            load=True
        )
        
        print(total_rewards, episode_collisions, episode_dones, when_dones)
    