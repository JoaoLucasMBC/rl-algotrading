"""
Evaluation script - compare RL agent to baseline strategies
"""

import yaml
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from environment import ExitEnv
from agent import DDQNAgent


def evaluate_rl_agent(agent, episodes, config):
    """Evaluate RL agent"""
    results = []
    lstm_warmup = config.get('lstm_warmup', 12)
    
    for episode_data in tqdm(episodes, desc="RL Agent"):
        env = ExitEnv(episode_data['data'], episode_data['entry_price'], warmup_steps=lstm_warmup)
        state = env.reset()
        hidden = agent.init_hidden()
        
        while True:
            action, hidden = agent.select_action(state, hidden, training=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            if done:
                results.append({
                    'return': info['pnl_pct'],
                    'days_held': info['days_held'],
                    'exit_price': info['exit_price']
                })
                break
    
    return results


def evaluate_fixed_profit(episodes, target=0.05):
    """Baseline: Exit at fixed profit target"""
    results = []
    
    for episode_data in tqdm(episodes, desc=f"Fixed {target*100:.0f}%"):
        entry_price = episode_data['entry_price']
        warmup_steps = episode_data.get('warmup_steps', 0)
        data = episode_data['data']
        
        # Start checking after warmup
        for i in range(warmup_steps, len(data)):
            row = data.iloc[i]
            pnl = (row['Close'] - entry_price) / entry_price
            
            # Check min hold days (3 days after warmup)
            days_held = i - warmup_steps
            if days_held < 3:
                continue
                
            if pnl >= target:
                results.append({
                    'return': pnl,
                    'days_held': days_held,
                    'exit_price': row['Close']
                })
                break
        else:
            # Exit at end
            final_price = data['Close'].iloc[-1]
            pnl = (final_price - entry_price) / entry_price
            results.append({
                'return': pnl,
                'days_held': len(data) - 1 - warmup_steps,
                'exit_price': final_price
            })
    
    return results


def evaluate_stop_loss(episodes, stop_loss=-0.02):
    """Baseline: Exit at stop loss"""
    results = []
    
    for episode_data in tqdm(episodes, desc=f"Stop Loss {stop_loss*100:.0f}%"):
        entry_price = episode_data['entry_price']
        warmup_steps = episode_data.get('warmup_steps', 0)
        data = episode_data['data']
        
        for i in range(warmup_steps, len(data)):
            row = data.iloc[i]
            pnl = (row['Close'] - entry_price) / entry_price
            
            # Check min hold days
            days_held = i - warmup_steps
            if days_held < 3:
                continue
            
            if pnl <= stop_loss:
                results.append({
                    'return': pnl,
                    'days_held': days_held,
                    'exit_price': row['Close']
                })
                break
        else:
            # Exit at end
            final_price = data['Close'].iloc[-1]
            pnl = (final_price - entry_price) / entry_price
            results.append({
                'return': pnl,
                'days_held': len(data) - 1 - warmup_steps,
                'exit_price': final_price
            })
    
    return results


def evaluate_trailing_stop(episodes, trail_pct=0.03):
    """Baseline: Trailing stop"""
    results = []
    
    for episode_data in tqdm(episodes, desc=f"Trailing {trail_pct*100:.0f}%"):
        entry_price = episode_data['entry_price']
        warmup_steps = episode_data.get('warmup_steps', 0)
        data = episode_data['data']
        
        # Initialize max price with entry price
        max_price = entry_price
        
        for i in range(warmup_steps, len(data)):
            row = data.iloc[i]
            current_price = row['Close']
            max_price = max(max_price, current_price)
            
            # Check min hold days
            days_held = i - warmup_steps
            if days_held < 3:
                continue
            
            # Exit if price drops trail_pct from peak
            if current_price < max_price * (1 - trail_pct):
                pnl = (current_price - entry_price) / entry_price
                results.append({
                    'return': pnl,
                    'days_held': days_held,
                    'exit_price': current_price
                })
                break
        else:
            # Exit at end
            final_price = data['Close'].iloc[-1]
            pnl = (final_price - entry_price) / entry_price
            results.append({
                'return': pnl,
                'days_held': len(data) - 1 - warmup_steps,
                'exit_price': final_price
            })
    
    return results


def evaluate_rsi_exit(episodes, rsi_threshold=70):
    """Baseline: Exit when RSI crosses above threshold"""
    results = []
    
    for episode_data in tqdm(episodes, desc=f"RSI Exit >{rsi_threshold}"):
        entry_price = episode_data['entry_price']
        warmup_steps = episode_data.get('warmup_steps', 0)
        data = episode_data['data']
        
        for i in range(warmup_steps, len(data)):
            row = data.iloc[i]
            
            # Check min hold days
            days_held = i - warmup_steps
            if days_held < 3:
                continue
            
            if row['RSI'] > rsi_threshold:
                pnl = (row['Close'] - entry_price) / entry_price
                results.append({
                    'return': pnl,
                    'days_held': days_held,
                    'exit_price': row['Close']
                })
                break
        else:
            # Exit at end
            final_price = data['Close'].iloc[-1]
            pnl = (final_price - entry_price) / entry_price
            results.append({
                'return': pnl,
                'days_held': len(data) - 1 - warmup_steps,
                'exit_price': final_price
            })
    
    return results


def print_results(results_dict):
    """Print comparison table"""
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print(f"{'Strategy':<20} {'Avg Return':>12} {'Win Rate':>10} {'Avg Days':>10} {'Sharpe':>10}")
    print("-"*80)
    
    for strategy, results in results_dict.items():
        returns = [r['return'] for r in results]
        days = [r['days_held'] for r in results]
        
        avg_return = np.mean(returns) * 100
        win_rate = np.mean([r > 0 for r in returns]) * 100
        avg_days = np.mean(days)
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        print(f"{strategy:<20} {avg_return:>11.2f}% {win_rate:>9.1f}% {avg_days:>9.1f} {sharpe:>10.2f}")
    
    print("="*80)


def plot_comparison(results_dict):
    """Plot comparison charts"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    strategies = list(results_dict.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
    
    # 1. Average returns
    ax = axes[0, 0]
    avg_returns = [np.mean([r['return'] for r in results_dict[s]]) * 100 for s in strategies]
    bars = ax.bar(strategies, avg_returns, color=colors)
    ax.set_ylabel('Avg Return (%)')
    ax.set_title('Average Return by Strategy')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Win rate
    ax = axes[0, 1]
    win_rates = [np.mean([r['return'] > 0 for r in results_dict[s]]) * 100 for s in strategies]
    ax.bar(strategies, win_rates, color=colors)
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Win Rate by Strategy')
    ax.axhline(50, color='black', linestyle='--', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Return distribution
    ax = axes[1, 0]
    for i, strategy in enumerate(strategies):
        returns = [r['return'] * 100 for r in results_dict[strategy]]
        ax.hist(returns, bins=30, alpha=0.5, label=strategy, color=colors[i])
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Return Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Days held
    ax = axes[1, 1]
    days_data = [[r['days_held'] for r in results_dict[s]] for s in strategies]
    bp = ax.boxplot(days_data, labels=strategies, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Days Held')
    ax.set_title('Holding Period Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('results/comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Comparison plots saved to results/comparison.png")


def main():
    print("="*60)
    print("EVALUATION - RL vs Baselines")
    print("="*60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test episodes
    print("\nLoading test episodes...")
    with open('data/processed/test_episodes.pkl', 'rb') as f:
        test_episodes = pickle.load(f)
    print(f"  Test: {len(test_episodes)} episodes")
    
    # Load RL agent
    print("\nLoading RL agent...")
    agent = DDQNAgent(state_size=19, action_size=2, config=config)
    agent.load('checkpoints/checkpoint_epoch_5.pth')
    agent.epsilon = 0  # No exploration during evaluation
    
    # Evaluate all strategies
    print("\nEvaluating strategies...")
    results_dict = {}
    
    results_dict['RL Agent'] = evaluate_rl_agent(agent, test_episodes, config)
    results_dict['Fixed 5%'] = evaluate_fixed_profit(test_episodes, target=0.05)
    results_dict['Stop Loss -2%'] = evaluate_stop_loss(test_episodes, stop_loss=-0.02)
    results_dict['Trailing 3%'] = evaluate_trailing_stop(test_episodes, trail_pct=0.03)
    results_dict['RSI Exit >70'] = evaluate_rsi_exit(test_episodes, rsi_threshold=70)
    
    # Print results
    print_results(results_dict)
    
    # Plot comparison
    plot_comparison(results_dict)
    
    # Save detailed results
    df_results = pd.DataFrame([
        {
            'strategy': strategy,
            'avg_return': np.mean([r['return'] for r in results]) * 100,
            'median_return': np.median([r['return'] for r in results]) * 100,
            'std_return': np.std([r['return'] for r in results]) * 100,
            'win_rate': np.mean([r['return'] > 0 for r in results]) * 100,
            'avg_days': np.mean([r['days_held'] for r in results]),
            'sharpe': np.mean([r['return'] for r in results]) / np.std([r['return'] for r in results])
        }
        for strategy, results in results_dict.items()
    ])
    
    df_results.to_csv('results/evaluation_results.csv', index=False)
    print("\n✓ Detailed results saved to results/evaluation_results.csv")
    
    print("\n" + "="*60)
    print("✓ EVALUATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
