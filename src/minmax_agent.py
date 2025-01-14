import math
from src.chess_engine import Move

PIECE_SCORES = {
    'K': 0,  # you might handle king differently
    'Q': 9,
    'R': 5,
    'B': 3,
    'N': 3,
    'p': 1
}

def evaluate_position(game_state):
    """Simple material-based evaluation."""
    score = 0
    for row in game_state.board:
        for piece in row:
            if piece != '--':
                sign = 1 if piece[0] == 'w' else -1
                score += sign * PIECE_SCORES.get(piece[1], 0)
    return score

def minimax(game_state, depth, alpha, beta, maximizing_player):
    if depth == 0:
        return evaluate_position(game_state), None

    valid_moves = game_state.get_valid_moves()
    if not valid_moves:
        return evaluate_position(game_state), None

    if maximizing_player:
        max_eval = -math.inf
        for move in valid_moves:
            if move.is_pawn_promotion:
                for promotion_piece in ['Q', 'R', 'B', 'N']:
                    move.promote_to = promotion_piece
                    move.is_pawn_promotion = False
                    game_state.make_move(move)
                    eval_score, _ = minimax(game_state, depth - 1, alpha, beta, False)
                    game_state.undo_move()
                    if eval_score > max_eval:
                        max_eval = eval_score
                        best_move = move
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
            else:
                game_state.make_move(move)
                eval_score, _ = minimax(game_state, depth - 1, alpha, beta, False)
                game_state.undo_move()
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in valid_moves:
            if move.is_pawn_promotion:
                for promotion_piece in ['Q', 'R', 'B', 'N']:
                    move.promote_to = promotion_piece
                    move.is_pawn_promotion = False
                    game_state.make_move(move)
                    eval_score, _ = minimax(game_state, depth - 1, alpha, beta, True)
                    game_state.undo_move()
                    if eval_score < min_eval:
                        min_eval = eval_score
                        best_move = move
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
            else:
                game_state.make_move(move)
                eval_score, _ = minimax(game_state, depth - 1, alpha, beta, True)
                game_state.undo_move()
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
        return min_eval, best_move

def get_best_move(game_state, depth=3):
    """Returns best move for the current player."""
    _, best_move = minimax(game_state, depth, -math.inf, math.inf, game_state.white_to_move)
    return best_move