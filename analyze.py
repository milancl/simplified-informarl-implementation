from sys import argv
import numpy as np
import pickle
import os.path as osp
import matplotlib.pyplot as plt

def extract_train(history):
    list = history["train"]
    
    s = [dict["steps"] for dict in list]
    act_loss = [dict["actor_loss"] for dict in list]
    cri_loss = [dict["critic_loss"] for dict in list]
    tot_rewards = [dict["total_episode_reward"] for dict in list]
    avg_rewards = [dict["average_episode_reward"] for dict in list]
    
    plt.title("Training Losses")
    plt.plot(s, act_loss, label='actor loss')
    plt.plot(s, cri_loss, label='critic loss')
    plt.legend()
    plt.show()
    
    plt.title("Training reward")
    plt.plot(s, avg_rewards, label='average episode reward') 
    plt.legend()
    plt.show()
    
def extract_eval(history):
    list = history["eval"]
    
    s = [dict["steps"] for dict in list]
    tot_rewards = [dict["total_episode_reward"] for dict in list]
    avg_rewards = [dict["average_episode_reward"] for dict in list]

    plt.title("Evaluation reward")
    plt.plot(s, avg_rewards, label='average episode reward') 
    plt.legend()
    plt.show()
        
    
if __name__ == "__main__":
    if len(argv) < 2:
        print("Run the program with '$ python analyze.py <run_name>'")
        exit(1)
    
    run_name = argv[1]
    run_dir = osp.join("runs", run_name)
    
    assert osp.exists(run_dir)
    
    with open(osp.join(run_dir, "history.pickle"), "rb") as f:
        history = pickle.load(f)
        
        
    extract_train(history)
    extract_eval(history)
        
    
    