import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import chess
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from tqdm import tqdm
import random
from datetime import datetime
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import math

MODEL_PATH = "alphazero_classic_agent"
BEST_MODEL_PATH = "best_alphazero_classic_agent"
CSV_PATH = "alphazero_classic_evaluation.csv"
TENSORBOARD_LOG = "./tensorboard/"
best_win_rate = 0

piece_values = {
    chess.PAWN: 0.1,
    chess.KNIGHT: 0.3,
    chess.BISHOP: 0.3,
    chess.ROOK: 0.5,
    chess.QUEEN: 0.9
}

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class PolicyValueNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # Initial convolution
        self.conv = nn.Conv2d(20, 256, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(256)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(5)])
        
        # Policy head
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2*8*8, features_dim)
        
        # Value head (for MCTS)
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8*8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Store value for MCTS
        self._value = None

    def forward(self, x):
        # Conversion et vérification des dimensions
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if x.shape[1] != 20:
            x = x.permute(0, 3, 1, 2)

        # Common trunk
        x = F.relu(self.bn(self.conv(x)))
        x = self.res_blocks(x)
        
        # Policy features
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2*8*8)
        policy = self.policy_fc(policy)
        
        # Value head (for MCTS)
        self._value = F.relu(self.value_bn(self.value_conv(x)))
        self._value = self._value.view(-1, 8*8)
        self._value = F.relu(self.value_fc1(self._value))
        self._value = torch.tanh(self.value_fc2(self._value))
        
        return policy  # Only return policy logits for SB3

    def get_value(self, x):
        self.forward(x)  # Compute value
        return self._value

class ClassicAlphaZeroEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.board = chess.Board()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(20,8,8), dtype=np.float32)
        self.action_space = spaces.Discrete(4672)
        self.last_material = self._material_balance()
        self.move_history = deque(maxlen=8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        self.last_material = self._material_balance()
        self.move_history.clear()
        return self._get_observation(), {}

    def step(self, action):
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return self._get_observation(), 0, True, False, {}
        
        move = legal_moves[action % len(legal_moves)]
        self.board.push(move)
        self.move_history.append(move)
        
        reward = self._get_reward()
        done = self.board.is_game_over()
        obs = self._get_observation()
        self.last_material = self._material_balance()
        
        return obs, reward, done, False, {}

    def _get_observation(self):
        board_matrix = np.zeros((20,8,8), dtype=np.float32)
        
        # Piece positions (channels 0-11)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                channel = (piece.piece_type-1) + (0 if piece.color == chess.WHITE else 6)
                board_matrix[channel, row, col] = 1
        
        # Additional information (channels 12-19)
        board_matrix[12] = 1 if self.board.turn == chess.WHITE else 0  # Color to move
        board_matrix[13] = 1 if self.board.has_kingside_castling_rights(chess.WHITE) else 0
        board_matrix[14] = 1 if self.board.has_queenside_castling_rights(chess.WHITE) else 0
        board_matrix[15] = 1 if self.board.has_kingside_castling_rights(chess.BLACK) else 0
        board_matrix[16] = 1 if self.board.has_queenside_castling_rights(chess.BLACK) else 0
        board_matrix[17] = self.board.fullmove_number / 100  # Move count
        board_matrix[18] = self.board.halfmove_clock / 50    # 50-move rule
        board_matrix[19] = 1 if self.board.is_repetition(2) else 0  # Repetition
        
        return board_matrix

    def _material_balance(self):
        white_score, black_score = 0, 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.piece_type in piece_values:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_score += value
                else:
                    black_score += value
        return white_score - black_score

    def _get_reward(self):
        reward = 0
        new_balance = self._material_balance()
        delta = new_balance - self.last_material
        reward += delta
        
        if self.board.is_check():
            reward += 0.2 if self.board.turn == chess.BLACK else -0.2
        
        if self.board.is_checkmate():
            mate_reward = 10 * (1 + new_balance if self.board.turn == chess.BLACK else 1 - new_balance)
            reward += mate_reward if self.board.turn == chess.BLACK else -mate_reward
        elif self.board.is_stalemate():
            penalty = -1 * (1 - abs(new_balance))
            reward += penalty
        elif self.board.is_insufficient_material():
            reward -= 0.5
        elif self.board.can_claim_fifty_moves():
            reward -= 0.5
        elif self.board.is_repetition(2):
            reward -= 0.5 if new_balance <= 0 else 0.2
        
        return reward

class MCTSNode:
    def __init__(self, parent=None, move=None, prior=0):
        self.parent = parent
        self.move = move
        self.children = []
        self.visit_count = 0
        self.total_value = 0
        self.prior = prior
    
    def is_expanded(self):
        return len(self.children) > 0
    
    def get_value(self):
        return self.total_value / (self.visit_count + 1e-6)
    
    def ucb_score(self, exploration_weight=1.0):
        if self.visit_count == 0:
            return float('inf')
        return self.get_value() + exploration_weight * self.prior * math.sqrt(self.parent.visit_count) / (self.visit_count + 1)

def run_mcts(env, model, root_node, simulations=200):
    for _ in range(simulations):
        node = root_node
        sim_env = env.__class__()
        sim_env.board = env.board.copy()
        
        # Selection
        while node.is_expanded():
            node = max(node.children, key=lambda n: n.ucb_score())
            sim_env.board.push(node.move)
        
        # Expansion
        if not sim_env.board.is_game_over():
            legal_moves = list(sim_env.board.legal_moves)
            obs = sim_env._get_observation()
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            
            with torch.no_grad():
                # Get policy and value from model
                features = model.policy.extract_features(obs_tensor)
                policy_logits = model.policy.mlp_extractor.policy_net(features)
                value = model.policy.mlp_extractor.value_net(features)
                
                # Convert to probabilities
                policy_probs = F.softmax(policy_logits, dim=1).numpy().flatten()
            
            for move in legal_moves:
                move_idx = legal_moves.index(move)
                prior = policy_probs[move_idx % len(legal_moves)]
                node.children.append(MCTSNode(parent=node, move=move, prior=prior))
            
            node = random.choices(node.children, weights=[c.prior for c in node.children])[0]
            sim_env.board.push(node.move)
        
        # Evaluation
        if sim_env.board.is_game_over():
            value = 1.0 if sim_env.board.result() == "1-0" else -1.0 if sim_env.board.result() == "0-1" else 0.0
        else:
            value = value.item()
        
        # Backpropagation
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value  # Switch perspective
            node = node.parent

def select_move_with_mcts(env, model, simulations=200, temperature=1.0):
    legal_moves = list(env.board.legal_moves)
    if not legal_moves:
        return random.randint(0, env.action_space.n - 1)
    
    root_node = MCTSNode()
    run_mcts(env, model, root_node, simulations)
    
    visit_counts = np.array([child.visit_count for child in root_node.children])
    
    if temperature == 0:
        best_move = root_node.children[np.argmax(visit_counts)].move
    else:
        visit_probs = visit_counts ** (1/temperature)
        visit_probs /= visit_probs.sum()
        best_move = root_node.children[np.random.choice(len(visit_probs), p=visit_probs)].move
    
    return legal_moves.index(best_move)

def save_evaluation_to_csv(steps, wins, draws, losses, win_rate):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_line = [now, steps, wins, draws, losses, round(win_rate, 4)]
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["date", "steps", "wins", "draws", "losses", "win_rate"])
        writer.writerow(new_line)

if __name__ == "__main__":
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    
    total_cycles = 100
    steps_per_cycle = 100_000
    eval_games = 100
    
    env = DummyVecEnv([lambda: ClassicAlphaZeroEnv() for _ in range(4)])
    
    policy_kwargs = dict(
        features_extractor_class=PolicyValueNetwork,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[dict(pi=[256], vf=[256])]  # Architecture séparée pour policy et value
    )
    
    if os.path.exists(BEST_MODEL_PATH + ".zip"):
        print("Loading best agent...")
        model = PPO.load(BEST_MODEL_PATH, env=env)
    else:
        print("Creating new PPO agent...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=5e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            tensorboard_log=TENSORBOARD_LOG
        )
    
    for cycle in range(total_cycles):
        print(f"Cycle {cycle+1}/{total_cycles} - Training {steps_per_cycle} steps")
        model.learn(
            total_timesteps=steps_per_cycle,
            reset_num_timesteps=False,
            tb_log_name="PPO"
        )
        
        # Evaluation with MCTS
        wins, draws, losses = 0, 0, 0
        for _ in tqdm(range(eval_games), desc="Evaluating"):
            test_env = ClassicAlphaZeroEnv()
            obs, _ = test_env.reset()
            done = False
            
            while not done:
                temperature = 1.0 if test_env.board.fullmove_number < 10 else 0.1
                action = select_move_with_mcts(test_env, model, simulations=200, temperature=temperature)
                obs, reward, done, _, _ = test_env.step(action)
            
            result = test_env.board.result()
            if result == "1-0":
                wins += 1
            elif result == "0-1":
                losses += 1
            else:
                draws += 1
        
        total_steps = (cycle+1) * steps_per_cycle
        win_rate = wins / eval_games
        print(f"Results: {wins}W {draws}D {losses}L | Win rate: {win_rate:.2%}")
        
        save_evaluation_to_csv(total_steps, wins, draws, losses, win_rate)
        
        if win_rate > best_win_rate:
            print(f"New best agent saved (win rate: {win_rate:.2%})")
            model.save(BEST_MODEL_PATH)
            best_win_rate = win_rate
    
    print("Training complete.")
    print(f"Best win rate achieved: {best_win_rate:.2%}")