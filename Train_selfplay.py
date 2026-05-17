import os
import numpy as np
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import json
from tqdm import tqdm
from datetime import datetime
from collections import deque
import math
import seaborn as sns

# Configuration
MODEL_PATH = "alphazero_optimized_agent"
BEST_MODEL_PATH = "best_alphazero_optimized_agent"
best_win_rate = 0

piece_values = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

class TrainingMonitor:
    def __init__(self):
        self.history = {
            'iteration': [],
            'total_loss': [], 'policy_loss': [], 'value_loss': [],
            'win_rate': [], 'draw_rate': [], 'loss_rate': [],
            'policy_accuracy': [], 'entropy': [],
            'game_lengths': [], 'mcts_simulations': [],
            'learning_rate': [], 'temperature': []
        }
        os.makedirs('training_plots', exist_ok=True)
    
    def update(self, iteration, losses, results, game_data, learning_rate=None, temperature=None, mcts_analysis=None):
        wins, draws, losses_count = results
        total_games = wins + draws + losses_count
        
        self.history['iteration'].append(iteration)
        self.history['total_loss'].append(losses['total'])
        self.history['policy_loss'].append(losses['policy'])
        self.history['value_loss'].append(losses['value'])
        self.history['win_rate'].append(wins / total_games if total_games > 0 else 0)
        self.history['draw_rate'].append(draws / total_games if total_games > 0 else 0)
        self.history['loss_rate'].append(losses_count / total_games if total_games > 0 else 0)
        
        if learning_rate is not None:
            self.history['learning_rate'].append(learning_rate)
        if temperature is not None:
            self.history['temperature'].append(temperature)
        
        # Calcul de l'accuracy policy
        if game_data:
            correct = 0
            total = 0
            for data in game_data:
                action = data['action']
                policy = data['mcts_policy']
                if len(policy) > 0:
                    predicted_action = np.argmax(policy)
                    if predicted_action == action:
                        correct += 1
                    total += 1
            
            if total > 0:
                self.history['policy_accuracy'].append(correct / total)
            else:
                self.history['policy_accuracy'].append(0)
            
            # Calcul de l'entropie
            entropies = []
            for data in game_data:
                policy = data['mcts_policy']
                if len(policy) > 0:
                    policy = np.clip(policy, 1e-10, 1.0)
                    entropy = -np.sum(policy * np.log(policy))
                    entropies.append(entropy)
            self.history['entropy'].append(np.mean(entropies) if entropies else 0)
        
        # Métriques MCTS
        if mcts_analysis:
            self.history['mcts_simulations'].append(np.mean([a['total_simulations'] for a in mcts_analysis]))
            self.history['game_lengths'].append(np.mean([a['game_length'] for a in mcts_analysis]))
        
        self.save_progress()
        
    def save_progress(self):
        with open('training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

def plot_training_progress(monitor):
    if len(monitor.history['iteration']) < 2:
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    iteration = monitor.history['iteration']
    
    # 1. Évolution des Losses
    axes[0,0].plot(iteration, monitor.history['total_loss'], label='Total Loss', linewidth=2, color='red')
    axes[0,0].plot(iteration, monitor.history['policy_loss'], label='Policy Loss', linewidth=2, color='blue')
    axes[0,0].plot(iteration, monitor.history['value_loss'], label='Value Loss', linewidth=2, color='green')
    axes[0,0].set_title('Évolution des Losses', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Itération')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Taux de victoires
    axes[0,1].plot(iteration, monitor.history['win_rate'], label='Victoires', linewidth=2, color='green')
    axes[0,1].plot(iteration, monitor.history['draw_rate'], label='Nulles', linewidth=2, color='orange')
    axes[0,1].plot(iteration, monitor.history['loss_rate'], label='Défaites', linewidth=2, color='red')
    axes[0,1].set_title('Résultats des Évaluations', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Itération')
    axes[0,1].set_ylabel('Taux')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim(0, 1)
    
    # 3. Accuracy policy
    if monitor.history['policy_accuracy']:
        axes[0,2].plot(iteration[-len(monitor.history['policy_accuracy']):], 
                      monitor.history['policy_accuracy'], 
                      label='Accuracy Policy', linewidth=2, color='purple')
        axes[0,2].set_title('Accuracy Policy', fontsize=14, fontweight='bold')
        axes[0,2].set_xlabel('Itération')
        axes[0,2].set_ylabel('Accuracy')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        axes[0,2].set_ylim(0, 1)
    
    # 4. Entropie des politiques
    if monitor.history['entropy']:
        axes[1,0].plot(iteration[-len(monitor.history['entropy']):], 
                      monitor.history['entropy'], 
                      label='Entropie Policy', linewidth=2, color='brown')
        axes[1,0].set_title('Entropie des Politiques', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Itération')
        axes[1,0].set_ylabel('Entropie')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # 5. Learning Rate
    if monitor.history['learning_rate']:
        axes[1,1].plot(iteration[-len(monitor.history['learning_rate']):], 
                      monitor.history['learning_rate'], 
                      label='Learning Rate', linewidth=2, color='blue')
        axes[1,1].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Itération')
        axes[1,1].set_ylabel('LR')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_yscale('log')
    
    # 6. Température
    if monitor.history['temperature']:
        axes[1,2].plot(iteration[-len(monitor.history['temperature']):], 
                      monitor.history['temperature'], 
                      label='Température', linewidth=2, color='orange')
        axes[1,2].set_title('Évolution de la Température', fontsize=14, fontweight='bold')
        axes[1,2].set_xlabel('Itération')
        axes[1,2].set_ylabel('Température')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
    
    # 7. Simulations MCTS
    if monitor.history['mcts_simulations']:
        axes[2,0].plot(iteration[-len(monitor.history['mcts_simulations']):], 
                      monitor.history['mcts_simulations'], 
                      label='Simulations MCTS', linewidth=2, color='teal')
        axes[2,0].set_title('Simulations MCTS Moyennes', fontsize=14, fontweight='bold')
        axes[2,0].set_xlabel('Itération')
        axes[2,0].set_ylabel('Nombre de Simulations')
        axes[2,0].legend()
        axes[2,0].grid(True, alpha=0.3)
    
    # 8. Longueur des parties
    if monitor.history['game_lengths']:
        axes[2,1].plot(iteration[-len(monitor.history['game_lengths']):], 
                      monitor.history['game_lengths'], 
                      label='Longueur Parties', linewidth=2, color='magenta')
        axes[2,1].set_title('Longueur Moyenne des Parties', fontsize=14, fontweight='bold')
        axes[2,1].set_xlabel('Itération')
        axes[2,1].set_ylabel('Nombre de Coups')
        axes[2,1].legend()
        axes[2,1].grid(True, alpha=0.3)
    
    # 9. Heatmap des métriques récentes
    metrics_to_plot = ['win_rate', 'policy_accuracy', 'total_loss']
    data = []
    labels = []
    
    for metric in metrics_to_plot:
        if monitor.history[metric]:
            values = monitor.history[metric][-8:]  # 8 dernières itérations
            if len(values) > 0:
                if metric == 'total_loss':
                    values = [-v for v in values]  # Inverser pour la visualisation
                data.append(values)
                labels.append(metric)
    
    if data:
        axes[2,2].remove()
        ax_heatmap = fig.add_subplot(3, 3, 9)
        sns.heatmap(data, annot=True, fmt='.3f', 
                   xticklabels=range(len(iteration)-len(data[0])+1, len(iteration)+1),
                   yticklabels=labels, cmap='RdYlGn', ax=ax_heatmap)
        ax_heatmap.set_title('Heatmap des Métriques (8 dernières itérations)')
    
    plt.tight_layout()
    plt.savefig('training_plots/training_progress.png', dpi=150, bbox_inches='tight')
    plt.close()

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class AlphaZeroPolicyNetwork(nn.Module):
    def __init__(self, action_space_size=4672):
        super().__init__()
        
        # Input: 20 channels (pieces + metadata)
        self.conv_input = nn.Conv2d(20, 256, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(256)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(5)])
        
        # Policy head
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2*8*8, action_space_size)
        
        # Value head
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8*8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if x.shape[1] != 20:
            x = x.permute(0, 3, 1, 2)

        # Input convolution
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual towers
        x = self.res_blocks(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.contiguous().view(-1, 2*8*8)
        policy_logits = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.contiguous().view(-1, 8*8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        # Assurer la bonne forme de la valeur
        if value.dim() == 0:
            value = value.unsqueeze(0)
        
        return policy_logits, value.squeeze()

class ClassicAlphaZeroEnv:
    def __init__(self):
        self.board = chess.Board()
        self.last_material = self._material_balance()

    def reset(self, seed=None, options=None):
        self.board.reset()
        self.last_material = self._material_balance()
        return self._get_observation(), {}

    def step(self, action):
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return self._get_observation(), 0, True, False, {}
        
        # Sélection du mouvement avec gestion des index invalides
        move_idx = action % len(legal_moves)
        move = legal_moves[move_idx]
        self.board.push(move)
        
        reward = self._get_reward()
        done = self.board.is_game_over()
        obs = self._get_observation()
        self.last_material = self._material_balance()
        
        return obs, reward, done, False, {}

    def _get_observation(self):
        board_matrix = np.zeros((20, 8, 8), dtype=np.float32)
        
        # Encodage des pièces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row, col = 7 - square // 8, square % 8  # Orientation échiquier standard
                piece_type = piece.piece_type - 1  # 0-5
                color_offset = 0 if piece.color == chess.WHITE else 6
                channel = piece_type + color_offset
                board_matrix[channel, row, col] = 1
        
        # Métadonnées
        board_matrix[12] = 1 if self.board.turn == chess.WHITE else 0
        board_matrix[13] = 1 if self.board.has_kingside_castling_rights(chess.WHITE) else 0
        board_matrix[14] = 1 if self.board.has_queenside_castling_rights(chess.WHITE) else 0
        board_matrix[15] = 1 if self.board.has_kingside_castling_rights(chess.BLACK) else 0
        board_matrix[16] = 1 if self.board.has_queenside_castling_rights(chess.BLACK) else 0
        board_matrix[17] = self.board.fullmove_number / 100.0
        board_matrix[18] = self.board.halfmove_clock / 50.0
        board_matrix[19] = 1 if self.board.is_repetition(2) else 0
        
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
        if self.board.is_checkmate():
            return 10.0 if self.board.turn == chess.BLACK else -10.0
        
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0.0
        
        # Récompense basée sur le matériel
        material_reward = self._material_balance() * 0.01
        
        # Petit bonus pour les échecs
        check_bonus = 0.1 if self.board.is_check() else 0.0
        
        return material_reward + check_bonus

class MCTSNode:
    def __init__(self, prior=0, parent=None):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.is_expanded = False
    
    def expand(self, action_priors):
        for action, prior in action_priors:
            self.children[action] = MCTSNode(prior=prior, parent=self)
        self.is_expanded = True
    
    def select_child(self, exploration_weight=1.0):
        return max(self.children.items(),
                  key=lambda item: item[1].ucb_score(exploration_weight))
    
    def ucb_score(self, exploration_weight):
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.total_value / self.visit_count
        exploration = exploration_weight * self.prior * \
                     math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        return exploitation + exploration

def run_mcts(env, model, simulations):
    root = MCTSNode()
    
    for _ in range(simulations):
        node = root
        sim_env = ClassicAlphaZeroEnv()
        sim_env.board = env.board.copy()
        
        # SELECTION
        while node.is_expanded and node.children:
            action, node = node.select_child()
            sim_env.step(action)
        
        # EXPANSION
        if not sim_env.board.is_game_over():
            legal_moves = list(sim_env.board.legal_moves)
            obs = sim_env._get_observation()
            
            with torch.no_grad():
                policy_logits, value = model(torch.FloatTensor(obs).unsqueeze(0))
                policy_probs = F.softmax(policy_logits, dim=1).squeeze().numpy()
            
            # Filtrer les probabilités pour les coups légaux
            action_priors = []
            for i, move in enumerate(legal_moves):
                prior = policy_probs[i % len(policy_probs)]
                action_priors.append((i, prior))
            
            node.expand(action_priors)
        
        # EVALUATION
        if sim_env.board.is_game_over():
            if sim_env.board.is_checkmate():
                value = -1.0  # Le joueur qui vient de jouer a perdu
            else:
                value = 0.0
        else:
            with torch.no_grad():
                obs = sim_env._get_observation()
                _, value = model(torch.FloatTensor(obs).unsqueeze(0))
                value = value.item()
        
        # BACKPROPAGATION
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value  # Alterner la perspective
            node = node.parent
    
    return root

def select_move_with_mcts(env, model, simulations, temperature):
    legal_moves = list(env.board.legal_moves)
    if not legal_moves:
        return 0
    
    root = run_mcts(env, model, simulations)
    
    visit_counts = np.array([child.visit_count for child in root.children.values()])
    
    if temperature == 0:
        # Mode exploitation pure
        best_idx = np.argmax(visit_counts)
    else:
        # Mode exploration
        visit_probs = visit_counts ** (1.0 / temperature)
        visit_probs /= np.sum(visit_probs)
        best_idx = np.random.choice(len(visit_probs), p=visit_probs)
    
    return list(root.children.keys())[best_idx]

def analyze_mcts_performance(env, model, num_positions=5):
    """Analyse détaillée des performances MCTS"""
    analyses = []
    
    for i in range(num_positions):
        test_env = ClassicAlphaZeroEnv()
        test_env.reset()
        
        # Jouer quelques coups aléatoires pour avoir une position intéressante
        for _ in range(np.random.randint(5, 15)):
            legal_moves = list(test_env.board.legal_moves)
            if legal_moves and not test_env.board.is_game_over():
                test_env.step(np.random.randint(len(legal_moves)))
            else:
                break
        
        legal_moves = list(test_env.board.legal_moves)
        if not legal_moves:
            continue
            
        root = run_mcts(test_env, model, simulations=25)
        
        visit_counts = [child.visit_count for child in root.children.values()]
        analyses.append({
            'game_length': len(legal_moves),
            'total_simulations': sum(visit_counts),
            'max_visits': max(visit_counts) if visit_counts else 0,
            'branching_factor': len(legal_moves),
            'position_complexity': len(legal_moves) * (sum(visit_counts) / len(visit_counts) if visit_counts else 0)
        })
    
    return analyses

def play_self_game(model, simulations_per_move, temperature):
    env = ClassicAlphaZeroEnv()
    obs, _ = env.reset()
    game_data = []
    move_count = 0
    
    while not env.board.is_game_over() and move_count < 150:  # Limite raisonnable
        current_obs = obs.copy()
        legal_moves = list(env.board.legal_moves)
        
        if not legal_moves:
            break
        
        # Obtenir la politique MCTS
        root = run_mcts(env, model, simulations_per_move)
        
        # Créer la distribution de probabilité
        mcts_policy = np.zeros(len(legal_moves))
        for i, child in root.children.items():
            if i < len(mcts_policy):
                mcts_policy[i] = child.visit_count
        
        total_visits = np.sum(mcts_policy)
        if total_visits > 0:
            mcts_policy /= total_visits
        else:
            mcts_policy = np.ones(len(legal_moves)) / len(legal_moves)
        
        # Sélectionner l'action
        action = select_move_with_mcts(env, model, simulations_per_move, temperature)
        
        # Jouer le coup
        next_obs, reward, done, _, _ = env.step(action)
        
        game_data.append({
            'observation': current_obs,
            'mcts_policy': mcts_policy,
            'action': action,
            'value_target': 0.0,
            'move_number': move_count
        })
        
        obs = next_obs
        move_count += 1
    
    # Déterminer le résultat final
    if env.board.is_checkmate():
        final_value = 1.0 if env.board.turn == chess.BLACK else -1.0
    else:
        final_value = 0.0
    
    # Assigner les valeurs cibles
    for i, data in enumerate(game_data):
        perspective = 1.0 if i % 2 == 0 else -1.0
        data['value_target'] = final_value * perspective
    
    return game_data

def train_on_self_play_data(model, self_play_data, epochs, learning_rate):
    if not self_play_data or len(self_play_data) == 0:
        print("    Aucune donnée d'entraînement")
        return {'total': 0, 'policy': 0, 'value': 0}
    
    try:
        # Préparer les données
        observations = []
        policy_targets = []
        value_targets = []
        
        for data in self_play_data:
            # Observation
            obs = data['observation']
            if obs is not None:
                observations.append(obs)
            
            # Policy target - conversion robuste vers 4672
            policy = data['mcts_policy']
            if isinstance(policy, np.ndarray):
                if policy.shape[0] == 4672:
                    policy_targets.append(policy)
                else:
                    # Conversion depuis les coups légaux vers 4672
                    policy_4672 = np.zeros(4672)
                    for i, prob in enumerate(policy):
                        if i < 4672:
                            policy_4672[i] = prob
                    policy_targets.append(policy_4672)
            else:
                policy_targets.append(np.zeros(4672))
            
            # Value target
            value_target = data['value_target']
            value_targets.append(value_target)
        
        # Vérifier que nous avons des données
        if len(observations) == 0:
            print("    Aucune observation valide")
            return {'total': 0, 'policy': 0, 'value': 0}
        
        # Convertir en tensors
        obs_tensor = torch.FloatTensor(np.array(observations))
        policy_target_tensor = torch.FloatTensor(np.array(policy_targets))
        value_target_tensor = torch.FloatTensor(np.array(value_targets))
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        model.train()
        losses = {'total': 0, 'policy': 0, 'value': 0}
        
        print(f"    Entraînement sur {len(observations)} échantillons (lr={learning_rate:.6f})...")
        
        for epoch in range(epochs):
            # Forward pass
            policy_logits, value_pred = model(obs_tensor)
            
            # Policy loss
            policy_loss = F.cross_entropy(policy_logits, policy_target_tensor)
            
            # Value loss - gestion des dimensions
            value_pred = value_pred.view(-1, 1)
            value_target = value_target_tensor.view(-1, 1)
            
            # Ajuster les dimensions si nécessaire
            min_len = min(value_pred.size(0), value_target.size(0))
            value_pred = value_pred[:min_len]
            value_target = value_target[:min_len]
            
            value_loss = F.mse_loss(value_pred, value_target)
            
            # Total loss
            total_loss = policy_loss + value_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Stocker la dernière loss
            if epoch == epochs - 1:
                losses = {
                    'total': total_loss.item(),
                    'policy': policy_loss.item(),
                    'value': value_loss.item()
                }
            
            print(f"      Epoch {epoch+1}/{epochs}: Loss={total_loss.item():.4f}")
        
        return losses
        
    except Exception as e:
        print(f"    Erreur pendant l'entraînement: {e}")
        return {'total': 0, 'policy': 0, 'value': 0}

def print_training_summary(iteration, losses, results, monitor, learning_rate, temperature):
    wins, draws, losses_count = results
    total_games = wins + draws + losses_count
    win_rate = wins / total_games if total_games > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"ITÉRATION {iteration} - RÉSULTATS DÉTAILLÉS")
    print(f"{'='*70}")
    
    print(f"PARAMÈTRES:")
    print(f"   Learning Rate: {learning_rate:.6f} | Température: {temperature:.3f}")
    
    print(f"LOSSES:")
    print(f"   Total: {losses['total']:.4f} | Policy: {losses['policy']:.4f} | Value: {losses['value']:.4f}")
    
    print(f"RÉSULTATS: {wins}V {draws}N {losses_count}D")
    print(f"   Taux de victoire: {win_rate:.2%}")
    
    if monitor.history['policy_accuracy']:
        acc = monitor.history['policy_accuracy'][-1]
        print(f"   Accuracy Policy: {acc:.2%}")
    
    if monitor.history['entropy']:
        entropy = monitor.history['entropy'][-1]
        print(f"   Entropie Policy: {entropy:.3f}")
    
    if monitor.history['mcts_simulations']:
        sims = monitor.history['mcts_simulations'][-1]
        print(f"   Simulations MCTS moyennes: {sims:.1f}")
    
    print(f"{'='*70}")

if __name__ == "__main__":
    os.makedirs('training_plots', exist_ok=True)
    
    # PARAMÈTRES OPTIMISÉS
    total_iterations = 100
    games_per_iteration = 2      # 2 parties pour plus de stabilité
    mcts_simulations = 30        # Plus de simulations pour de meilleures politiques
    training_epochs = 3          # Plus d'epochs pour un meilleur apprentissage
    initial_temperature = 1.5
    evaluation_temperature = 0.1  # Température fixe pour l'évaluation
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device: {device}")
    
    model = AlphaZeroPolicyNetwork()
    model.to(device)
    
    # Initialisation des poids
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)  # Petit biais positif
    
    model.apply(init_weights)
    
    monitor = TrainingMonitor()
    
    print("ALPHAZERO OPTIMISÉ - DÉMARRAGE DE L'ENTRAÎNEMENT")
    print(f"Configurations:")
    print(f"   Itérations totales: {total_iterations}")
    print(f"   Parties par itération: {games_per_iteration}")
    print(f"   Simulations MCTS: {mcts_simulations}")
    print(f"   Epochs d'entraînement: {training_epochs}")
    print(f"   Température initiale: {initial_temperature}")
    print(f"   Device: {device}")
    
    for iteration in range(total_iterations):
        print(f"\n{'='*60}")
        print(f"ITÉRATION {iteration+1}/{total_iterations}")
        print(f"{'='*60}")
        
        # Calcul de la température et learning rate adaptatifs
        current_temperature = max(0.1, initial_temperature * (0.97 ** iteration))
        learning_rate = 0.001 * (0.95 ** (iteration // 15))  # Réduction tous les 15 itérations
        
        print(f"Température: {current_temperature:.3f} | Learning Rate: {learning_rate:.6f}")
        
        # Phase 1: Self-Play
        print("Phase 1: Self-Play...")
        all_self_play_data = []
        total_moves = 0
        
        for game_idx in range(games_per_iteration):
            game_data = play_self_game(model, mcts_simulations, current_temperature)
            all_self_play_data.extend(game_data)
            total_moves += len(game_data)
            print(f"  Partie {game_idx+1}: {len(game_data)} coups")
        
        print(f"Total données d'entraînement: {len(all_self_play_data)} positions")
        
        # Phase 2: Entraînement
        print("Phase 2: Entraînement du réseau...")
        losses = train_on_self_play_data(model, all_self_play_data, training_epochs, learning_rate)
        
        # Phase 3: Évaluation
        print("Phase 3: Évaluation...")
        wins, draws, losses_count = 0, 0, 0
        evaluation_moves = []
        
        for eval_game in range(5):  # 5 parties d'évaluation
            env = ClassicAlphaZeroEnv()
            env.reset()
            move_count = 0
            
            while not env.board.is_game_over() and move_count < 150:
                action = select_move_with_mcts(env, model, mcts_simulations, evaluation_temperature)
                env.step(action)
                move_count += 1
            
            result = env.board.result()
            if result == "1-0":
                wins += 1
            elif result == "0-1":
                losses_count += 1
            else:
                draws += 1
            
            evaluation_moves.append(move_count)
        
        # Analyse MCTS périodique
        mcts_analysis = None
        if (iteration + 1) % 10 == 0:
            print("Analyse détaillée MCTS...")
            mcts_analysis = analyze_mcts_performance(ClassicAlphaZeroEnv(), model, num_positions=5)
            if mcts_analysis:
                avg_simulations = np.mean([a['total_simulations'] for a in mcts_analysis])
                avg_complexity = np.mean([a['position_complexity'] for a in mcts_analysis])
                print(f"  Simulations moyennes: {avg_simulations:.1f}")
                print(f"  Complexité moyenne: {avg_complexity:.1f}")
        
        # Mise à jour du monitoring
        results = (wins, draws, losses_count)
        monitor.update(iteration + 1, losses, results, all_self_play_data, 
                      learning_rate, current_temperature, mcts_analysis)
        
        # Visualisation périodique
        if (iteration + 1) % 5 == 0:
            plot_training_progress(monitor)
        
        # Affichage du résumé
        print_training_summary(iteration + 1, losses, results, monitor, learning_rate, current_temperature)
        
        win_rate = wins / (wins + draws + losses_count) if (wins + draws + losses_count) > 0 else 0
        
        # Sauvegarde du meilleur modèle
        if win_rate > best_win_rate:
            print(f"NOUVEAU MEILLEUR MODÈLE! Taux de victoire: {win_rate:.2%}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'iteration': iteration + 1,
                'win_rate': win_rate,
                'losses': losses,
                'history': monitor.history,
                'config': {
                    'games_per_iteration': games_per_iteration,
                    'mcts_simulations': mcts_simulations,
                    'training_epochs': training_epochs
                }
            }, BEST_MODEL_PATH + ".pth")
            best_win_rate = win_rate
        
        # Checkpoints périodiques
        if (iteration + 1) % 20 == 0:
            checkpoint_path = f"checkpoint_iteration_{iteration+1}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'iteration': iteration + 1,
                'win_rate': win_rate,
                'monitor_history': monitor.history,
                'learning_rate': learning_rate,
                'temperature': current_temperature
            }, checkpoint_path)
            print(f"Checkpoint sauvegardé: {checkpoint_path}")
    
    # Sauvegarde finale
    torch.save(model.state_dict(), MODEL_PATH + "_final.pth")
    
    print(f"\n{'='*70}")
    print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print(f"{'='*70}")
    print(f"Meilleur taux de victoire atteint: {best_win_rate:.2%}")
    print(f"Données sauvegardées dans:")
    print(f"   training_history.json")
    print(f"   training_plots/training_progress.png")
    print(f"   {BEST_MODEL_PATH}.pth (meilleur modèle)")
    print(f"   {MODEL_PATH}_final.pth (modèle final)")
    
    # Graphique final
    plot_training_progress(monitor)
    print(f"   training_plots/training_progress.png (graphique final)")
