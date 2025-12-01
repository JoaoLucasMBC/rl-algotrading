"""
Training script for DDQN agent
"""

import yaml
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from environment import ExitEnv
from agent import DDQNAgent


def train_agent(agent, train_episodes, val_episodes, config):
    """
    Train DDQN agent on exit strategy
    """
    epochs = config.get('epochs', 15)
    save_every = config.get('save_every', 1)
    lstm_warmup = config.get('lstm_warmup', 12)
    
    best_val_reward = -np.inf
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        # Training
        epoch_rewards = []
        epoch_losses = []
        epoch_exit_days = []
        
        # Shuffle training episodes
        np.random.shuffle(train_episodes)
        
        for episode_data in tqdm(train_episodes, desc="Training"):
            env = ExitEnv(episode_data['data'], episode_data['entry_price'], warmup_steps=lstm_warmup)
            state = env.reset()
            episode_reward = 0
            episode_transitions = []
            
            # Initialize hidden state for LSTM
            hidden = agent.init_hidden()
            
            # Episode-level exploration (Paper's method)
            # "With probability epsilon, episode is random and a uniformly random day to stop is selected"
            is_random_episode = np.random.random() < agent.epsilon
            random_stop_step = -1
            if is_random_episode:
                # Can only stop after warmup
                # max_steps is total length. Valid stop indices: [warmup, max_steps-1]
                if env.max_steps > lstm_warmup:
                    random_stop_step = np.random.randint(lstm_warmup, env.max_steps)
            
            while True:
                # Select action
                if is_random_episode:
                    # Force random policy
                    if env.current_step < random_stop_step:
                        action = 0 # HOLD
                    elif env.current_step == random_stop_step:
                        action = 1 # SELL
                    else:
                        action = 1 # Should have stopped already
                    
                    # Still pass through network to update hidden state
                    _, hidden = agent.select_action(state, hidden, training=False)
                else:
                    # Use learned policy (Greedy)
                    # "Otherwise, the policy learned so far is used throughout the episode"
                    action, hidden = agent.select_action(state, hidden, training=False)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                # Only store if we are past warmup (or should we store warmup too? 
                # Paper says "learning only from the part of the trajectory that precedes stopping time".
                # But LSTM needs history. We should store everything so LSTM can process it.)
                episode_transitions.append((state, action, reward, next_state, done))
                
                episode_reward += reward
                state = next_state
                
                if done:
                    epoch_exit_days.append(info['days_held'])
                    break
            
            # Store full episode
            agent.store_episode(episode_transitions)
            
            # Train on batch of episodes
            loss = agent.train_step()
            
            if loss is not None:
                epoch_losses.append(loss)
            
            epoch_rewards.append(episode_reward)
        
        avg_train_reward = np.mean(epoch_rewards)
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        avg_exit_day = np.mean(epoch_exit_days)
        
        # Validation
        val_rewards = []
        val_exit_days = []
        
        for episode_data in tqdm(val_episodes, desc="Validation"):
            env = ExitEnv(episode_data['data'], episode_data['entry_price'], warmup_steps=lstm_warmup)
            state = env.reset()
            episode_reward = 0
            hidden = agent.init_hidden()
            
            while True:
                # Greedy action for validation
                action, hidden = agent.select_action(state, hidden, training=False)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state
                if done:
                    val_exit_days.append(info['days_held'])
                    break
            
            val_rewards.append(episode_reward)
        
        avg_val_reward = np.mean(val_rewards)
        avg_val_exit = np.mean(val_exit_days)
        
        # Print stats
        print(f"\nResults:")
        print(f"  Train Reward: {avg_train_reward*100:6.2f}%  |  Avg Exit Day: {avg_exit_day:.1f}")
        print(f"  Val Reward:   {avg_val_reward*100:6.2f}%  |  Avg Exit Day: {avg_val_exit:.1f}")
        print(f"  Loss: {avg_loss:.4f}  |  Epsilon: {agent.epsilon:.3f}")
        
        # Save stats
        agent.episode_rewards.append({
            'epoch': epoch,
            'train_reward': avg_train_reward,
            'val_reward': avg_val_reward,
            'loss': avg_loss
        })
        
        # Save best model
        if avg_val_reward > best_val_reward:
            best_val_reward = avg_val_reward
            agent.save('checkpoints/best_model.pth')
            print(f"  ✓ New best model! Val reward: {best_val_reward*100:.2f}%")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            agent.save(f'checkpoints/checkpoint_epoch_{epoch+1}.pth')
        
        # Early stopping (less aggressive)
        if epoch > 5 and avg_val_reward < best_val_reward * 0.5:
            print("\n⚠️  Early stopping - validation performance degrading significantly")
            break
    
    return agent


def plot_training_curves(agent):
    """Plot training curves"""
    stats = agent.episode_rewards
    
    epochs = [s['epoch'] for s in stats]
    train_rewards = [s['train_reward']*100 for s in stats]
    val_rewards = [s['val_reward']*100 for s in stats]
    losses = [s['loss'] for s in stats]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Rewards
    ax1.plot(epochs, train_rewards, 'b-', label='Train', marker='o')
    ax1.plot(epochs, val_rewards, 'r-', label='Val', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Avg Return (%)')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(epochs, losses, 'g-', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=150, bbox_inches='tight')
    print("\n✓ Training curves saved to results/training_curves.png")


def main():
    print("="*60)
    print("TRAINING DDQN AGENT")
    print("="*60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load episodes
    print("\nLoading episodes...")
    with open('data/processed/train_episodes.pkl', 'rb') as f:
        train_episodes = pickle.load(f)
    with open('data/processed/val_episodes.pkl', 'rb') as f:
        val_episodes = pickle.load(f)
    
    print(f"  Train: {len(train_episodes)} episodes")
    print(f"  Val:   {len(val_episodes)} episodes")
    
    # Create agent
    print("\nInitializing agent...")
    agent = DDQNAgent(state_size=19, action_size=2, config=config)
    
    # Train
    print("\nStarting training...")
    agent = train_agent(agent, train_episodes, val_episodes, config)
    
    # Plot results
    plot_training_curves(agent)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
