import gymnasium as gym
from gymnasium import spaces
import numpy as np
import chess
import chess.svg
import pygame
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from collections import deque
from typing import Optional, List

# Paramètres Pygame
TILE_SIZE = 80
BOARD_SIZE = TILE_SIZE * 8
MARGIN = 50
PANEL_HEIGHT = 40
WINDOW_WIDTH = BOARD_SIZE + 2 * MARGIN
WINDOW_HEIGHT = BOARD_SIZE + 2 * MARGIN + PANEL_HEIGHT
WHITE = (238, 238, 210)
BLACK = (118, 150, 86)
HIGHLIGHT = (186, 202, 68, 150)
LEGAL_MOVE_COLOR = (200, 0, 0, 150)
LAST_MOVE_COLOR = (255, 215, 0, 150)
PANEL_COLOR = (240, 240, 240)
TEXT_COLOR = (0, 0, 0)
FONT_SIZE = 24
PIECE_IMAGES = {}

# Valeurs des pièces
piece_values = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

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
        self.last_move = None

    def reset(self, seed=None, options=None):
        self.board.reset()
        self.last_material = self._material_balance()
        self.last_move = None
        return self._get_observation(), {}

    def step(self, action):
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return self._get_observation(), 0, True, False, {}
        
        move_idx = action % len(legal_moves)
        move = legal_moves[move_idx]
        self.board.push(move)
        self.last_move = move
        
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
                row, col = 7 - square // 8, square % 8
                piece_type = piece.piece_type - 1
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

def run_mcts(env, model, simulations=50):
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

def select_move_with_mcts(env, model, simulations=50, temperature=0.1):
    """Sélectionne un coup avec MCTS (température basse pour exploitation)"""
    legal_moves = list(env.board.legal_moves)
    if not legal_moves:
        return 0
    
    root = run_mcts(env, model, simulations)
    
    visit_counts = np.array([child.visit_count for child in root.children.values()])
    
    if temperature == 0:
        # Mode exploitation pure
        best_idx = np.argmax(visit_counts)
    else:
        # Mode exploration contrôlée
        visit_probs = visit_counts ** (1.0 / temperature)
        visit_probs /= np.sum(visit_probs)
        best_idx = np.random.choice(len(visit_probs), p=visit_probs)
    
    return list(root.children.keys())[best_idx]

def load_piece_images():
    """Charge les images des pièces"""
    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    colors = ['w', 'b']
    
    if not os.path.exists('Assets'):
        os.makedirs('Assets')
        print("Le dossier 'Assets' a été créé. Veuillez y placer les images des pièces.")
        return False
    
    for color in colors:
        for piece in pieces:
            try:
                image_path = f'Assets/{color}{piece}.png'
                if os.path.exists(image_path):
                    PIECE_IMAGES[color + piece] = pygame.transform.scale(
                        pygame.image.load(image_path),
                        (TILE_SIZE, TILE_SIZE)
                    )
                else:
                    print(f"Image manquante: {image_path}")
                    return False
            except Exception as e:
                print(f"Erreur lors du chargement de l'image {color}{piece}.png: {e}")
                return False
    return True

def draw_board(screen, board, selected_square, legal_moves, last_move, game_status, turn, ai_thinking=False):
    """Dessine l'échiquier et le panneau d'information"""
    # Dessiner le fond
    screen.fill((200, 200, 200))
    
    # Dessiner le panneau d'information
    font = pygame.font.SysFont("Arial", FONT_SIZE)
    status_text = f"Tour: {'Blancs' if turn == chess.WHITE else 'Noirs'} | Statut: {game_status}"
    if ai_thinking:
        status_text += " | IA réfléchit..."
    
    text_surface = font.render(status_text, True, TEXT_COLOR)
    screen.blit(text_surface, (MARGIN, MARGIN // 2 - FONT_SIZE // 2))
    
    # Afficher le côté que l'IA joue
    ai_side_text = font.render("IA: Noirs", True, TEXT_COLOR)
    screen.blit(ai_side_text, (WINDOW_WIDTH - MARGIN - 100, MARGIN // 2 - FONT_SIZE // 2))
    
    # Dessiner l'échiquier
    board_rect = pygame.Rect(MARGIN, MARGIN + PANEL_HEIGHT, BOARD_SIZE, BOARD_SIZE)
    pygame.draw.rect(screen, (150, 150, 150), board_rect)
    
    for row in range(8):
        for col in range(8):
            square = chess.square(col, 7 - row)
            rect = pygame.Rect(
                MARGIN + col * TILE_SIZE,
                MARGIN + PANEL_HEIGHT + row * TILE_SIZE,
                TILE_SIZE, TILE_SIZE
            )
            
            # Couleur de la case
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, rect)
            
            # Surbrillance du dernier mouvement
            if last_move and (square == last_move.from_square or square == last_move.to_square):
                highlight = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                highlight.fill(LAST_MOVE_COLOR)
                screen.blit(highlight, rect)
            
            # Surbrillance de la case sélectionnée
            if selected_square == square:
                highlight = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                highlight.fill(HIGHLIGHT)
                screen.blit(highlight, rect)
            
            # Marqueurs des mouvements légaux
            if square in legal_moves:
                center = (
                    MARGIN + col * TILE_SIZE + TILE_SIZE // 2,
                    MARGIN + PANEL_HEIGHT + row * TILE_SIZE + TILE_SIZE // 2
                )
                pygame.draw.circle(screen, LEGAL_MOVE_COLOR, center, 10)
            
            # Pièces
            piece = board.piece_at(square)
            if piece:
                piece_color = 'w' if piece.color == chess.WHITE else 'b'
                piece_symbol = piece.symbol().lower()
                screen.blit(
                    PIECE_IMAGES[piece_color + piece_symbol],
                    (MARGIN + col * TILE_SIZE, MARGIN + PANEL_HEIGHT + row * TILE_SIZE)
                )

def get_game_status(board):
    """Retourne le statut actuel du jeu"""
    if board.is_checkmate():
        return "Échec et mat"
    elif board.is_stalemate():
        return "Pat"
    elif board.is_insufficient_material():
        return "Matériel insuffisant"
    elif board.is_check():
        return "Échec"
    elif board.is_game_over():
        return "Partie terminée"
    else:
        return "En cours"

def load_model(model_path):
    """Charge le modèle AlphaZero entraîné"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Chargement du modèle sur {device}")
    
    model = AlphaZeroPolicyNetwork()
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Modèle chargé depuis {model_path}")
            print(f"Meilleur win rate: {checkpoint.get('win_rate', 0):.2%}")
            print(f"Itération: {checkpoint.get('iteration', 'N/A')}")
        else:
            model.load_state_dict(checkpoint)
            print("Modèle chargé (format simple)")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        print("Utilisation d'un modèle aléatoire")
    
    model.to(device)
    model.eval()
    return model, device

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Jeu d'échecs contre AlphaZero")
    clock = pygame.time.Clock()
    
    # Charger le modèle AlphaZero
    model_paths = [
        "best_alphazero_optimized_agent.pth",
        "best_alphazero_classic_agent.pth",
        "alphazero_optimized_agent_final.pth"
    ]
    
    model = None
    device = None
    
    for path in model_paths:
        if os.path.exists(path):
            model, device = load_model(path)
            break
    
    if model is None:
        print("Aucun modèle trouvé. Utilisation d'un agent aléatoire.")
        model = AlphaZeroPolicyNetwork()
        device = torch.device("cpu")
        model.to(device)
        model.eval()
    
    # Initialiser l'environnement
    env = ClassicAlphaZeroEnv()
    env.reset()
    
    # Charger les images des pièces
    if not load_piece_images():
        print("Impossible de charger les images des pièces. Le programme va quitter.")
        return
    
    # Variables de jeu
    running = True
    selected_square = None
    legal_moves = []
    game_status = "En cours"
    ai_thinking = False
    
    # Option : qui joue les blancs ?
    player_is_white = True  # True = joueur joue les blancs, False = joueur joue les noirs
    
    while running:
        # Mettre à jour le statut du jeu
        game_status = get_game_status(env.board)
        
        # Dessiner l'échiquier
        draw_board(screen, env.board, selected_square, legal_moves, 
                  env.last_move, game_status, env.board.turn, ai_thinking)
        pygame.display.flip()
        
        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN and game_status == "En cours":
                x, y = pygame.mouse.get_pos()
                
                # Vérifier que le clic est dans l'échiquier
                if (MARGIN <= x < MARGIN + BOARD_SIZE and 
                    MARGIN + PANEL_HEIGHT <= y < MARGIN + PANEL_HEIGHT + BOARD_SIZE):
                    
                    col = (x - MARGIN) // TILE_SIZE
                    row = 7 - (y - MARGIN - PANEL_HEIGHT) // TILE_SIZE
                    square = chess.square(col, row)
                    
                    # Déterminer si c'est le tour du joueur
                    player_turn = (player_is_white and env.board.turn == chess.WHITE) or \
                                 (not player_is_white and env.board.turn == chess.BLACK)
                    
                    if player_turn:
                        if selected_square is None:
                            # Sélectionner une pièce du joueur
                            piece = env.board.piece_at(square)
                            if piece and piece.color == (chess.WHITE if player_is_white else chess.BLACK):
                                selected_square = square
                                legal_moves = [move.to_square for move in env.board.legal_moves 
                                             if move.from_square == selected_square]
                        else:
                            # Déplacer la pièce sélectionnée
                            move = chess.Move(selected_square, square)
                            if move in env.board.legal_moves:
                                env.board.push(move)
                                env.last_move = move
                                game_status = get_game_status(env.board)
                                
                                # Réinitialiser la sélection
                                selected_square = None
                                legal_moves = []
                                
                                # Vérifier si c'est le tour de l'IA
                                ai_turn = (player_is_white and env.board.turn == chess.BLACK) or \
                                         (not player_is_white and env.board.turn == chess.WHITE)
                                
                                if not env.board.is_game_over() and ai_turn:
                                    ai_thinking = True
                                    draw_board(screen, env.board, selected_square, legal_moves,
                                              env.last_move, game_status, env.board.turn, ai_thinking)
                                    pygame.display.flip()
                                    
                                    # L'IA joue son coup
                                    action = select_move_with_mcts(env, model, simulations=50, temperature=0.1)
                                    env.step(action)
                                    game_status = get_game_status(env.board)
                                    
                                    ai_thinking = False
                            else:
                                # Clic sur une autre pièce du joueur
                                piece = env.board.piece_at(square)
                                if piece and piece.color == (chess.WHITE if player_is_white else chess.BLACK):
                                    selected_square = square
                                    legal_moves = [move.to_square for move in env.board.legal_moves 
                                                 if move.from_square == selected_square]
                                else:
                                    selected_square = None
                                    legal_moves = []
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Touche R pour réinitialiser
                    env.reset()
                    selected_square = None
                    legal_moves = []
                    game_status = "En cours"
                elif event.key == pygame.K_u:  # Touche U pour annuler le dernier coup
                    if len(env.board.move_stack) > 0:
                        env.board.pop()
                        if len(env.board.move_stack) > 0:
                            env.last_move = env.board.move_stack[-1]
                        else:
                            env.last_move = None
                        game_status = get_game_status(env.board)
                elif event.key == pygame.K_s:  # Touche S pour changer de côté
                    player_is_white = not player_is_white
                    print(f"Le joueur joue maintenant les {'blancs' if player_is_white else 'noirs'}")
                elif event.key == pygame.K_a:  # Touche A pour que l'IA joue un coup (utile pour debug)
                    if not env.board.is_game_over():
                        ai_thinking = True
                        draw_board(screen, env.board, selected_square, legal_moves,
                                  env.last_move, game_status, env.board.turn, ai_thinking)
                        pygame.display.flip()
                        
                        action = select_move_with_mcts(env, model, simulations=50, temperature=0.1)
                        env.step(action)
                        game_status = get_game_status(env.board)
                        
                        ai_thinking = False
        
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()