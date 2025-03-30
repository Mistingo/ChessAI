import gymnasium as gym
from gymnasium import spaces
import numpy as np
import chess
import chess.svg
import pygame
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from typing import Optional, Tuple, List

# Paramètres Pygame
TILE_SIZE = 80
BOARD_SIZE = TILE_SIZE * 8
MARGIN = 50
PANEL_HEIGHT = 40
WINDOW_WIDTH = BOARD_SIZE + 2 * MARGIN
WINDOW_HEIGHT = BOARD_SIZE + 2 * MARGIN + PANEL_HEIGHT
WHITE = (238, 238, 210)
BLACK = (118, 150, 86)
HIGHLIGHT = (186, 202, 68, 150)  # Ajout de transparence
LEGAL_MOVE_COLOR = (200, 0, 0, 150)
LAST_MOVE_COLOR = (255, 215, 0, 150)
PANEL_COLOR = (240, 240, 240)
TEXT_COLOR = (0, 0, 0)
FONT_SIZE = 24
PIECE_IMAGES = {}

class ChessEnv(gym.Env):
    """Environnement d'échecs pour l'apprentissage par renforcement."""
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.render_mode = "human"
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.float32)
        self.action_space = spaces.Discrete(4672)
        self.last_move = None
        self.move_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        self.last_move = None
        self.move_history = []
        return self._get_observation(), {}

    def step(self, action):
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return self._get_observation(), 0, True, False, {}
        
        action = action % len(legal_moves)
        move = legal_moves[action]
        self.board.push(move)
        self.last_move = move
        self.move_history.append(move.uci())
        
        reward = self._get_reward()
        done = self.board.is_game_over()
        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        board_matrix = np.zeros((8, 8, 12), dtype=np.float32)
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                row, col = divmod(i, 8)
                board_matrix[row, col, piece.piece_type - 1] = 1 if piece.color == chess.WHITE else -1
        return board_matrix

    def _get_reward(self):
        if self.board.is_checkmate():
            return 1 if self.board.turn == chess.BLACK else -1
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0
        else:
            # Récompense basée sur la valeur matérielle
            piece_values = {
                chess.PAWN: 1,
                chess.KNIGHT: 3,
                chess.BISHOP: 3,
                chess.ROOK: 5,
                chess.QUEEN: 9,
                chess.KING: 0
            }
            white_material = 0
            black_material = 0
            
            for square in chess.SQUARES:
                piece = self.board.piece_at(square)
                if piece:
                    value = piece_values[piece.piece_type]
                    if piece.color == chess.WHITE:
                        white_material += value
                    else:
                        black_material += value
            
            material_diff = white_material - black_material
            return material_diff * 0.01

    def render(self, mode="human"):
        print(self.board)

    def close(self):
        pass

def load_piece_images():
    """Charge les images des pièces avec gestion des erreurs."""
    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    colors = ['w', 'b']
    
    # Vérifier si le dossier images existe
    if not os.path.exists('images'):
        os.makedirs('images')
        print("Le dossier 'images' a été créé. Veuillez y placer les images des pièces.")
        return False
    
    for color in colors:
        for piece in pieces:
            try:
                image_path = f'images/{color}{piece}.png'
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

def draw_board(screen: pygame.Surface, board: chess.Board, selected_square: Optional[int], 
               legal_moves: List[int], last_move: Optional[chess.Move], 
               game_status: str, turn: chess.Color) -> None:
    """Dessine l'échiquier, les pièces et le panneau d'information."""
    # Dessiner le fond
    screen.fill((200, 200, 200))
    
    # Dessiner le panneau d'information
    font = pygame.font.SysFont("Arial", FONT_SIZE)
    status_text = f"Tour: {'Blancs' if turn == chess.WHITE else 'Noirs'} | Statut: {game_status}"
    text_surface = font.render(status_text, True, TEXT_COLOR)
    screen.blit(text_surface, (MARGIN, MARGIN // 2 - FONT_SIZE // 2))
    
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

def get_game_status(board: chess.Board) -> str:
    """Retourne le statut actuel du jeu."""
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

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Jeu d'échecs IA")
    clock = pygame.time.Clock()
    
    # Charger l'agent IA
    model_path = input("Entrez le chemin vers le modèle IA (sans extension): ") or "best_chess_agent"
    try:
        model = PPO.load(model_path)
        print(f"Modèle {model_path} chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        print("Utilisation d'un agent aléatoire à la place.")
        model = None
    
    env = ChessEnv()
    if not load_piece_images():
        print("Impossible de charger les images des pièces. Le programme va quitter.")
        return
    
    running = True
    selected_square = None
    legal_moves = []
    game_status = "En cours"
    
    while running:
        game_status = get_game_status(env.board)
        draw_board(screen, env.board, selected_square, legal_moves, env.last_move, game_status, env.board.turn)
        pygame.display.flip()
        
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
                    
                    if selected_square is None:
                        # Sélectionner une pièce du joueur (blanc)
                        if (env.board.piece_at(square) and 
                            env.board.piece_at(square).color == chess.WHITE and 
                            env.board.turn == chess.WHITE):
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
                            
                            # Tour de l'IA (noir)
                            if not env.board.is_game_over() and env.board.turn == chess.BLACK:
                                obs = env._get_observation()
                                action = model.predict(obs)[0] if model else np.random.randint(env.action_space.n)
                                env.step(action)
                                game_status = get_game_status(env.board)
                        
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
    
    pygame.quit()

if __name__ == "__main__":
    main()