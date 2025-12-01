"""
LSTM Q-Network with Dueling Architecture
Following the paper's architecture: 3-layer LSTM with 512 cells
"""

import torch
import torch.nn as nn


class LSTMQNetwork(nn.Module):
    """
    LSTM-based Q-Network with Dueling Architecture
    Same as paper: 512 cells per layer
    """
    
    def __init__(self, input_size=19, hidden_size=512, num_layers=3, num_actions=2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Dueling architecture
        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # Advantage stream: A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_actions)
        )
    
    def forward(self, x, hidden=None):
        """
        Forward pass
        x: (batch, seq_len, input_size) or (batch, input_size)
        Returns: q_values (batch, num_actions), hidden_state
        """
        # Handle single state (add sequence dimension)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_size)
        
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)
        
        # We want to keep the sequence dimension for training
        # lstm_out: (batch, seq_len, hidden_size)
        
        # Dueling streams
        # Linear layers work on the last dimension, so they handle (batch, seq_len, hidden) fine
        value = self.value_stream(lstm_out)  # (batch, seq_len, 1)
        advantage = self.advantage_stream(lstm_out)  # (batch, seq_len, num_actions)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # Mean across the action dimension (last dimension)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values, hidden
