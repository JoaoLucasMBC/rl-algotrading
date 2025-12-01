"""
DDQN Agent with LSTM
Implements Double Deep Q-Learning from the paper
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from model import LSTMQNetwork
import os


class DDQNAgent:
    """
    Double DQN Agent with LSTM
    Following the paper's approach
    """
    
    def __init__(self, state_size=19, action_size=2, config=None):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.learning_rate = config.get('learning_rate', 0.0001)
        self.batch_size = config.get('batch_size', 64)
        self.memory_size = config.get('memory_size', 10000)
        self.target_update = config.get('target_update', 1)
        self.tau = config.get('tau', 0.001)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Networks
        hidden_size = config.get('hidden_size', 512)
        num_layers = config.get('num_layers', 3)
        
        self.policy_net = LSTMQNetwork(state_size, hidden_size, num_layers, action_size).to(self.device)
        self.target_net = LSTMQNetwork(state_size, hidden_size, num_layers, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Replay memory
        self.memory = deque(maxlen=self.memory_size)
        
        # Training stats
        self.steps = 0
        self.episode_rewards = []
        self.losses = []
    
    def init_hidden(self, batch_size=1):
        """Initialize hidden state for LSTM"""
        return None  # Let LSTM initialize to zeros
    
    def select_action(self, state, hidden=None, training=True):
        """
        Select action using epsilon-greedy
        Returns: action, next_hidden
        """
        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            action = random.randint(0, self.action_size - 1)
            # Need to pass state through network to update hidden state even if random action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
                _, next_hidden = self.policy_net(state_tensor, hidden)
            return action, next_hidden
        
        with torch.no_grad():
            # Add batch and sequence dimensions: (1, 1, input_size)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            q_values, next_hidden = self.policy_net(state_tensor, hidden)
            return q_values.argmax().item(), next_hidden
    
    def store_episode(self, episode_transitions):
        """Store full episode in replay memory"""
        # episode_transitions is a list of (state, action, reward, next_state, done)
        self.memory.append(episode_transitions)
    
    def train_step(self):
        """
        Train on batches of full episodes
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch of episodes
        episodes = random.sample(self.memory, self.batch_size)

        # print if there is any NaN in the episodes
        for ep in episodes:
            for state, action, reward, next_state, done in ep:
                if np.isnan(state).any():
                    print("NaN in state")
                if np.isnan(action).any():
                    print("NaN in action")
                if np.isnan(reward).any():
                    print("NaN in reward")
                if np.isnan(next_state).any():
                    print("NaN in next_state")
                if np.isnan(done).any():
                    print("NaN in done")
        
        # Prepare batch data
        # We need to pad episodes to the same length
        max_len = max(len(ep) for ep in episodes)
        
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        mask = []  # To mask out padding
        
        for ep in episodes:
            # Unzip transitions
            states, actions, rewards, next_states, dones = zip(*ep)
            
            # Pad to max_len
            pad_len = max_len - len(ep)
            
            # Pad with zeros (or last state)
            # States: (seq_len, input_size)
            s = np.array(states)
            s_pad = np.pad(s, ((0, pad_len), (0, 0)), mode='constant')
            batch_states.append(s_pad)
            
            # Actions: (seq_len,)
            a = np.array(actions)
            a_pad = np.pad(a, (0, pad_len), mode='constant')
            batch_actions.append(a_pad)
            
            # Rewards: (seq_len,)
            r = np.array(rewards)
            r_pad = np.pad(r, (0, pad_len), mode='constant')
            batch_rewards.append(r_pad)
            
            # Next States: (seq_len, input_size)
            ns = np.array(next_states)
            ns_pad = np.pad(ns, ((0, pad_len), (0, 0)), mode='constant')
            batch_next_states.append(ns_pad)
            
            # Dones: (seq_len,)
            d = np.array(dones)
            d_pad = np.pad(d, (0, pad_len), mode='constant', constant_values=1) # Pad dones with 1 (True)
            batch_dones.append(d_pad)
            
            # Mask: 1 for valid, 0 for padding
            m = np.concatenate([np.ones(len(ep)), np.zeros(pad_len)])
            mask.append(m)
            
        # Convert to tensors
        # Shape: (batch, seq_len, input_size)
        states = torch.FloatTensor(np.array(batch_states)).to(self.device)
        actions = torch.LongTensor(np.array(batch_actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(batch_rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(batch_next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(batch_dones)).to(self.device)
        mask = torch.FloatTensor(np.array(mask)).to(self.device)
        
        # Forward pass (Policy Net)
        # LSTM processes full sequence
        # q_values: (batch, seq_len, num_actions)
        # hidden states are handled internally by LSTM (initialized to 0)
        q_values, _ = self.policy_net(states)
        
        # Gather Q-values for taken actions
        # actions: (batch, seq_len) -> (batch, seq_len, 1)
        current_q = q_values.gather(2, actions.unsqueeze(2)).squeeze(2)
            
        # Double DQN Target
        with torch.no_grad():
            # Next Q from Policy Net (for action selection)
            next_q_policy, _ = self.policy_net(next_states)
            next_actions = next_q_policy.argmax(2)  # (batch, seq_len)
            
            # Next Q from Target Net (for value estimation)
            next_q_target, _ = self.target_net(next_states)
            # Gather
            next_q = next_q_target.gather(2, next_actions.unsqueeze(2)).squeeze(2)
            
            # Target calculation
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        # Loss with masking
        # Only compute loss for valid steps
        loss_elementwise = nn.SmoothL1Loss(reduction='none')(current_q, target_q)
        loss = (loss_elementwise * mask).sum() / (mask.sum() + 1e-8)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        # theta_target = tau * theta_local + (1 - tau) * theta_target
        if self.steps % self.target_update == 0:
            for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path):
        """Load model checkpoint"""
        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.losses = checkpoint.get('losses', [])
        print(f"✓ Model loaded from {path}")
