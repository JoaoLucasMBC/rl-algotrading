import matplotlib.pyplot as plt
from tbparse import SummaryReader
import os
import pandas as pd

log_dir = "./logs_dqn"

# Find the latest run directory or file
# The structure seems to be logs_dqn/events... or logs_dqn/DQN_X/events...
# We'll look for the most recent events file
def find_latest_log(directory):
    events_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "events.out.tfevents" in file:
                events_files.append(os.path.join(root, file))
    
    if not events_files:
        return None
    
    # Sort by modification time
    events_files.sort(key=os.path.getmtime, reverse=True)
    return events_files[0]

latest_log = find_latest_log(log_dir)
print(f"Reading logs from: {latest_log}")

if latest_log:
    reader = SummaryReader(os.path.dirname(latest_log))
    df = reader.scalars
    
    if df.empty:
        print("No scalar data found in logs.")
    else:
        # Filter for reward and loss
        # Common tags in SB3: 'rollout/ep_rew_mean', 'train/loss'
        
        rewards = df[df['tag'] == 'rollout/ep_rew_mean']
        losses = df[df['tag'] == 'train/loss']
        
        plt.figure(figsize=(12, 10))
        
        # Plot Reward
        plt.subplot(2, 1, 1)
        if not rewards.empty:
            plt.plot(rewards['step'], rewards['value'], label='Mean Episode Reward', color='blue')
            plt.title('Mean Episode Reward over Time')
            plt.xlabel('Timesteps')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No Reward Data Available', ha='center', va='center')
            print("No reward data found.")

        # Plot Loss
        plt.subplot(2, 1, 2)
        if not losses.empty:
            plt.plot(losses['step'], losses['value'], label='Training Loss', color='red')
            plt.title('Training Loss over Time')
            plt.xlabel('Timesteps')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No Loss Data Available', ha='center', va='center')
            print("No loss data found.")
            
        plt.tight_layout()
        plt.savefig('training_results.png')
        print("Plots saved to training_results.png")
        # plt.show() # Cannot show in headless env
else:
    print("No logs found.")
