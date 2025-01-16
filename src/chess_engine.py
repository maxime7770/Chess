"""
Storing the information abour the current state of the chess game
Determining the valid moves at the current state. (move log)
"""


from asyncio.format_helpers import _format_callback_source
from string import whitespace
import numpy as np


class GameState:
    
    def __init__(self):

        # board is an 8x8 2D list: characters represent the pieces (from white persective)
        self.board = [
            ['bR', 'bN', 'bB', 'bQ', 'bK', 'bB', 'bN', 'bR'],
            ['bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp'],
            ['--', '--', '--', '--', '--', '--', '--', '--'], 
            ['--', '--', '--', '--', '--', '--', '--', '--'],
            ['--', '--', '--', '--', '--', '--', '--', '--'],
            ['--', '--', '--', '--', '--', '--', '--', '--'],
            ['wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp'],
            ['wR', 'wN', 'wB', 'wQ', 'wK', 'wB', 'wN', 'wR'],
        ]
        self.move_functions = {'p': self.get_pawn_moves, 'R': self.get_rook_moves, 'N': self.get_knight_moves,
                               'B': self.get_bishop_moves, 'Q': self.get_queen_moves, 'K': self.get_king_moves}
        self.white_to_move = True
        self.in_check = False
        self.move_log = []
        self.white_king_location = (7, 4)
        self.black_king_location = (0, 4)
        self.pins = [] # list of tuples (row, col, x, y) of pieces that are pinned and direction of pin
        self.checks = [] # list of tuples (row, col) of pieces that are in check and direction of check
        self.enpassant_possible = ()   # coordinate of the possible square
        self.checkmate = False
        self.stalemate = False

    
    def make_move(self, move):
        ''' makes a move using the result from Move class
        '''
        self.board[move.start_row][move.start_col] = '--'
        self.board[move.end_row][move.end_col] = move.piece_moved
        self.move_log.append(move)
        self.white_to_move = not self.white_to_move 
        # update king location
        if move.piece_moved == 'wK':
            self.white_king_location = (move.end_row, move.end_col)
        if move.piece_moved == 'bK':
            self.black_king_location = (move.end_row, move.end_col)

        if move.is_pawn_promotion:
            # promoted = input('Promote to Q, B, R, or N: ')
            # while promoted not in ['Q', 'B', 'R', 'N']:
            #     promoted = input('Invalid input. Promote to Q, B, R, or N: ')
            promoted = 'Q'
            self.board[move.end_row][move.end_col] = move.piece_moved[0] + promoted

        if move.is_enpassant:
            self.board[move.start_row][move.end_col] = '--' # capture

        if move.piece_moved[1] == 'p' and abs(move.start_row - move.end_row) == 2: # 2 squares advance
            self.enpassant_possible = ((move.start_row + move.end_row) // 2, move.start_col)
        else:
            self.enpassant_possible = ()

    def create_initial_board(self):
        """Initialize the chess board to the starting position."""
        return [
            ['bR', 'bN', 'bB', 'bQ', 'bK', 'bB', 'bN', 'bR'],
            ['bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp'],
            ['--', '--', '--', '--', '--', '--', '--', '--'],
            ['--', '--', '--', '--', '--', '--', '--', '--'],
            ['--', '--', '--', '--', '--', '--', '--', '--'],
            ['--', '--', '--', '--', '--', '--', '--', '--'],
            ['wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp'],
            ['wR', 'wN', 'wB', 'wQ', 'wK', 'wB', 'wN', 'wR']
        ]
    
    def reset(self):
        """Reset the game to the initial state."""
        self.board = self.create_initial_board()
        self.move_log = []
        self.white_to_move = True
        self.white_king_location = (7, 4)
        self.black_king_location = (0, 4)
        self.enpassant_possible = ()

    def encode_move(self, move):
        """
        Encodes a Move object into a unique integer in [0..4095],
        where start and end squares each map to 0..63.
        """
        start_idx = move.start_row * 8 + move.start_col  # [0..63]
        end_idx   = move.end_row * 8 + move.end_col      # [0..63]
        return start_idx * 64 + end_idx  # [0..4095]

    def decode_move(self, action_index):
        """
        Decodes an action index in [0..4095] back into a Move object.
        """
        start_idx = action_index // 64   # [0..63]
        end_idx   = action_index % 64    # [0..63]
        start_row, start_col = divmod(start_idx, 8)
        end_row,   end_col   = divmod(end_idx, 8)
        return Move((start_row, start_col), (end_row, end_col), self.board)
                
    def get_state_representation(self):
        ''' Converts the board into a flattened 8x8 array.
        Each piece type is represented by a unique integer.
        Empty squares are 0.
        '''
        piece_to_int = {
            '--': 0,
            'wp': 1, 'wN': 2, 'wB': 3, 'wR': 4, 'wQ': 5, 'wK': 6,
            'bp': -1, 'bN': -2, 'bB': -3, 'bR': -4, 'bQ': -5, 'bK': -6
        }
        state = []
        for row in self.board:
            for piece in row:
                state.append(piece_to_int.get(piece, 0))
        return np.array(state, dtype=np.float32)
    

    def get_valid_move_indices(self):
        ''' Returns a list of action indices corresponding to all valid moves.
        Each action index uniquely identifies a move.
        '''
        valid_moves = self.get_valid_moves()
        move_indices = []
        for move in valid_moves:
            index = self.encode_move(move)  # Implement this encoding
            move_indices.append(index)
        return move_indices

    def get_move_from_action(self, action_index):
        """
        Converts an action index back to a Move object.
        """
        move = self.decode_move(action_index)  # Implement this decoding
        return move


    def undo_move(self):
        ''' undo the last move made 
        '''
        if len(self.move_log) != 0:
            move = self.move_log.pop()
            self.board[move.start_row][move.start_col] = move.piece_moved   
            self.board[move.end_row][move.end_col] = move.piece_captured
            self.white_to_move = not self.white_to_move  # switch back
            if move.piece_moved == 'wK':
                self.white_king_location = (move.start_row, move.start_col)
            if move.piece_moved == 'bK':
                self.black_king_location = (move.start_row, move.start_col)
            if move.is_enpassant:
                self.board[move.end_row][move.end_col] = '--'
                self.board[move.start_row][move.end_col] = move.piece_captured
                self.enpassant_possible = (move.end_row, move.end_col)
            if move.piece_moved[1] == 'p' and abs(move.start_row - move.end_row) == 2:
                self.enpassant_possible = ()

    def get_valid_moves(self):
        ''' all moves considering checks 
        '''
        moves = []
        self.in_check, self.pins, self.checks = self.pins_and_checks()
        if self.white_to_move:
            king_row, king_col = self.white_king_location[0], self.white_king_location[1]
        else:
            king_row, king_col = self.black_king_location[0], self.black_king_location[1]
        if self.in_check:
            if len(self.checks) == 1: # only 1 check: have to either block check or move king
                moves = self.get_all_possible_moves()
                check = self.checks[0]
                check_row, check_col = check[0], check[1]
                piece_checking = self.board[check_row][check_col]
                valid_squares = []
                if piece_checking[1] == 'N':  # you cannot block a knight, you have to take it 
                    valid_squares = [(check_row, check_col)]
                else:
                    for i in range(1, 8):
                        valid_square = (king_row + check[2] * i, king_col + check[3] * i)
                        valid_squares.append(valid_square)
                        if valid_square[0] == check_row and valid_square[1] == check_col: # break when you get to the checking piece
                            break
                for i in range(len(moves) - 1, -1, -1):  # go backwards while removing elements
                    if moves[i].piece_moved[1] != 'K':
                        if not (moves[i].end_row, moves[i].end_col) in valid_squares:
                            moves.remove(moves[i])
            else: # double check
                self.get_king_moves(king_row, king_col, moves)
        else: # no check
            moves = self.get_all_possible_moves()
        return moves


    def pins_and_checks(self):
        ''' check for pins or checks
        '''
        pins = []
        checks = []
        in_check = False
        if self.white_to_move:
            ennemy_color = 'b'
            ally_color = 'w'
            start_row, start_col = self.white_king_location[0], self.white_king_location[1]
        else:
            ennemy_color = 'w'
            ally_color = 'b'
            start_row, start_col = self.black_king_location[0], self.black_king_location[1]
        directions = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for j in range(len(directions)):
            d = directions[j]
            possible_pin = ()  # reset possible pins
            for i in range(1, 8):
                end_row, end_col = start_row + d[0] * i, start_col + d[1] * i  # direction
                if 0 <= end_row <= 7 and 0 <= end_col <= 7:
                    end_piece = self.board[end_row][end_col]
                    if end_piece[0] == ally_color and end_piece[1] != 'K':
                        if possible_pin == ():
                            possible_pin = (end_row, end_col, d[0], d[1])
                        else:  # means that there is a nearest pin
                            break 
                    elif end_piece[0] == ennemy_color:  # end_piece could represent a check 
                        piece = end_piece[1]
                        if (0 <= j <= 3 and piece == 'R') or (4 <= j <= 7 and piece == 'B') or \
                            (i == 1 and piece == 'p' and ((ennemy_color == 'w' and 6 <= j <= 7) or (ennemy_color == 'b' and 4 <= j <= 5))) or \
                                (piece == 'Q') or (i == 1 and piece == 'K'):
                                if possible_pin == ():  # no piece is blocking the check
                                    in_check = True
                                    checks.append((end_row, end_col, d[0], d[1]))
                                    break
                                else:  # there is a piece blocking the check   
                                    pins.append(possible_pin)
                                    break
                        else:   # no check
                            break
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        for move in knight_moves:
            end_row, end_col = start_row + move[0], start_col + move[1]
            if 0 <= end_row <= 7 and 0 <= end_col <= 7:
                end_piece = self.board[end_row][end_col]
                if end_piece[0] == ennemy_color and end_piece[1] == 'N':
                    in_check = True
                    checks.append((end_row, end_col, move[0], move[1]))
        return in_check, pins, checks
    

    def is_checkmate(self):
        """
        Determines if the current state is a checkmate.
        Returns True if the current player's king is in check and there are no valid moves.
        """
        if self.white_to_move:
            return self.square_under_attack(self.white_king_location[0], self.white_king_location[1])
        else:
            return self.square_under_attack(self.black_king_location[0], self.black_king_location[1])
    
    def is_stalemate(self):
        """
        Determines if the current state is a stalemate.
        Returns True if the current player is not in check but has no valid moves.
        """
        self.in_check, _, _ = self.pins_and_checks()  # Check if the current player is in check
        if not self.in_check:
            valid_moves = self.get_valid_moves()
            if len(valid_moves) == 0:  # No valid moves and not in check
                return True
        return 
    

    def square_under_attack(self, row, col):
        """
        Determine if enemy can attack the square row col
        """
        self.white_to_move = not self.white_to_move  # switch to opponent's point of view
        opponents_moves = self.get_all_possible_moves()
        self.white_to_move = not self.white_to_move
        for move in opponents_moves:
            if move.end_row == row and move.end_col == col:  # square is under attack
                return True
        return False
            
                         
    
    def get_all_possible_moves(self):
        ''' all moves without considering checks
        '''
        moves = []   # see the __eq__ method that change equality relation from default ('is') to == 
        for r in range(len(self.board)):
            for c in range(len(self.board[r])):
                turn = self.board[r][c][0]
                if (turn == 'w' and self.white_to_move) or (turn == 'b' and not self.white_to_move):
                    piece = self.board[r][c][1]
                    self.move_functions[piece](r, c, moves)   # appropriate move function for the piece 

        return moves

    

    def get_pawn_moves(self, r, c, moves):
        ''' get all the pawn possible moves for the pawn located at row, col and add them to the list moves
        '''
        piece_pinned = False
        pin_direction = ()
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piece_pinned = True
                pin_direction = (self.pins[i][2], self.pins[i][3])
                self.pins.remove(self.pins[i])
                break

        if self.white_to_move:   # white is playing

            if self.board[r-1][c] == '--':   # 1 square move
                if not piece_pinned or pin_direction == (-1, 0):
                    moves.append(Move((r, c), (r-1, c), self.board))
                    if r == 6 and self.board[r-2][c] == '--':  # 2 squares move in the beginning
                        moves.append(Move((r, c), (r-2, c), self.board))
                
            if c - 1 >= 0:  # capture to the left
                if self.board[r-1][c-1][0] == 'b':   # ennemy piece to capture 
                    if not piece_pinned or pin_direction == (-1, -1):
                        moves.append(Move((r, c), (r-1, c-1), self.board))
                elif (r-1, c-1) == self.enpassant_possible:
                    moves.append(Move((r, c), (r-1, c-1), self.board, is_enpassant=True))
            if c + 1 <= 7:  # capture to the right
                if self.board[r-1][c+1][0] == 'b':   # ennemy piece to capture 
                    if not piece_pinned or pin_direction == (-1, 1):
                        moves.append(Move((r, c), (r-1, c+1), self.board))
                elif (r-1, c+1) == self.enpassant_possible:
                    moves.append(Move((r, c), (r-1, c+1), self.board, is_enpassant=True))
            


        else:  # black is playing

            if self.board[r+1][c] == '--':   # 1 square move 
                if not piece_pinned or pin_direction == (1, 0):
                    moves.append(Move((r, c), (r+1, c), self.board))
                    if r == 1 and self.board[r+2][c] == '--':  # 2 squares move in the beginning
                        moves.append(Move((r, c), (r+2, c), self.board))
            
            if c - 1 >= 0:  # capture to the left (from white perspective...)
                if self.board[r+1][c-1][0] == 'w':   # ennemy piece to capture 
                    if not piece_pinned or pin_direction == (1, -1):
                        moves.append(Move((r, c), (r+1, c-1), self.board))
                elif (r+1, c-1) == self.enpassant_possible:
                    moves.append(Move((r, c), (r+1, c-1), self.board, is_enpassant=True))
            
            if c + 1 <= 7:  # capture to the right
                if self.board[r+1][c+1][0] == 'w':   # ennemy piece to capture 
                    if not piece_pinned or pin_direction == (1, 1):
                         moves.append(Move((r, c), (r+1, c+1), self.board))
                elif (r+1, c+1) == self.enpassant_possible:
                    moves.append(Move((r, c), (r+1, c+1), self.board, is_enpassant=True))



    def get_rook_moves(self, r, c, moves):
        ''' get all the rook possible moves for the rook located at row, col and add them to the list moves
        '''
        piece_pinned = False
        pin_direction = ()
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piece_pinned = True
                pin_direction = (self.pins[i][2], self.pins[i][3])
                if self.board[r][c][1] != 'Q':
                     self.pins.remove(self.pins[i])
                break

        if self.white_to_move:
            ennemy = 'b'
        else:
            ennemy = 'w'

        up, down, left, right = r - 1, r + 1, c - 1, c + 1

        if not piece_pinned or pin_direction == (-1, 0) or pin_direction == (1, 0):
            while up >= 0 and self.board[up][c] == '--':   # go up when you can
                moves.append(Move((r, c), (up, c), self.board))
                up -= 1
            if up >= 0 and self.board[up][c][0] == ennemy:    # capture
                moves.append(Move((r, c), (up, c), self.board))

        if not piece_pinned or pin_direction == (-1, 0) or pin_direction == (1, 0):
            while down <= 7 and self.board[down][c] == '--':   # go down when you can
                moves.append(Move((r, c), (down, c), self.board))
                down += 1
            if down <= 7 and self.board[down][c][0] == ennemy:    
                moves.append(Move((r, c), (down, c), self.board))
        
        if not piece_pinned or pin_direction == (0, 1) or pin_direction == (0, -1):
            while left >= 0 and self.board[r][left] == '--':   # go left when you can
                moves.append(Move((r, c), (r, left), self.board))
                left -= 1
            if left >= 0 and self.board[r][left][0] == ennemy:    
                moves.append(Move((r, c), (r, left), self.board))
        
        if not piece_pinned or pin_direction == (0, 1) or pin_direction == (0, -1):
            while right <= 7 and self.board[r][right] == '--':   # go right when you can
                moves.append(Move((r, c), (r, right), self.board))
                right += 1
            if right <= 7 and self.board[r][right][0] == ennemy:   
                moves.append(Move((r, c), (r, right), self.board))
        



    def get_knight_moves(self, r, c, moves):
        ''' get all the knight possible moves for the knight located at row, col and add them to the list moves
        '''
        piece_pinned = False
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piece_pinned = True
                self.pins.remove(self.pins[i])
                break

        if self.white_to_move:
            ally = 'w'
        else:
            ally = 'b'

        moves_knight = [(r + 2, c + 1), (r + 2, c - 1), (r - 2, c + 1), (r - 2, c - 1), 
                        (r + 1, c + 2), (r - 1, c + 2), (r + 1, c - 2), (r - 1, c - 2)]
        
        for (x, y) in moves_knight:
            if not piece_pinned:
                if 0 <= x <= 7 and 0 <= y <= 7 and self.board[x][y][0] != ally:
                    moves.append(Move((r, c), (x, y), self.board))



    def get_bishop_moves(self, r, c, moves):
        ''' get all the bishop possible moves for the bishop located at row, col and add them to the list moves
        '''
        piece_pinned = False
        pin_direction = ()
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piece_pinned = True
                pin_direction = (self.pins[i][2], self.pins[i][3])
                self.pins.remove(self.pins[i])
                break

        if self.white_to_move:
            ennemy = 'b'
        else:
            ennemy = 'w'

        up, left = r - 1, c - 1
        if not piece_pinned or pin_direction == (-1, -1) or pin_direction == (1, 1):
            while up >= 0 and c >= 0 and self.board[up][left] == '--':   # go along the top left diagonal
                moves.append(Move((r, c), (up, left), self.board))
                up -= 1
                left -= 1
            if up >= 0 and left >= 0 and self.board[up][left][0] == ennemy:    # capture
                moves.append(Move((r, c), (up, left), self.board))
        
        up, right = r - 1, c + 1
        if not piece_pinned or pin_direction == (-1, 1) or pin_direction == (1, -1):
            while up >= 0 and right <= 7 and self.board[up][right] == '--':   # go along the top right diagonal
                moves.append(Move((r, c), (up, right), self.board))
                up -= 1
                right += 1
            if up >= 0 and right <= 7 and self.board[up][right][0] == ennemy:    
                moves.append(Move((r, c), (up, right), self.board))
            
        down, left = r + 1, c - 1
        if not piece_pinned or pin_direction == (-1, 1) or pin_direction == (1, -1):
            while down <= 7 and left >= 0 and self.board[down][left] == '--':   # go along the bottom left diagonal
                moves.append(Move((r, c), (down, left), self.board))
                down += 1
                left -= 1
            if down <= 7 and left >= 0 and self.board[down][left][0] == ennemy:    
                moves.append(Move((r, c), (down, left), self.board))
            
        down, right = r + 1, c + 1
        if not piece_pinned or pin_direction == (-1, 0) or pin_direction == (1, 0):
            while down <= 7 and right <= 7 and self.board[down][right] == '--':   # go along the bottom right diagonal
                moves.append(Move((r, c), (down, right), self.board))
                down += 1
                right += 1
            if down <= 7 and right <= 7 and self.board[down][right][0] == ennemy:    
                moves.append(Move((r, c), (down, right), self.board))
        


    def get_queen_moves(self, r, c, moves):
        ''' get all the queen possible moves for the queen located at row, col and add them to the list moves
        '''
         # a queen can move like a rook and a bishop
        self.get_rook_moves(r, c, moves)
        self.get_bishop_moves(r, c, moves)
        


    def get_king_moves(self, r, c, moves):
        ''' get all the king possible moves for the king located at row, col and add them to the list moves
        '''
        # for now just look at all possible moves, not valid ones 

        if self.white_to_move:
            ally = 'w'
        else:
            ally = 'b'
        
        moves_king = [(r + 1, c + 1), (r + 1, c - 1), (r + 1, c), (r, c + 1),
                      (r, c - 1), (r - 1, c - 1), (r - 1, c), (r - 1, c + 1)]

        for (x, y) in moves_king:
            if 0 <= x <= 7 and 0 <= y <= 7 and self.board[x][y][0] != ally:
                if ally == 'w':
                    self.white_king_location = (x, y)
                else:
                    self.black_king_location = (x, y)
                in_check, _, _ = self.pins_and_checks()
                if not in_check:
                    moves.append(Move((r, c), (x, y), self.board))
                
                if ally == 'w':  # take king back to its previous location
                    self.white_king_location = (r, c)
                else:
                    self.black_king_location = (r, c)


  
        

    
class Move:

    ranks_to_row = {'1': 7, '2': 6, '3': 5, '4': 4, 
                    '5': 3, '6': 2, '7': 1, '8': 0}
    row_to_ranks = {v: k for k, v in ranks_to_row.items()}
    files_to_cols = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 
                    'e': 4, 'f': 5, 'g': 6, 'h': 7}
    cols_to_files = {v: k for k, v in files_to_cols.items()}

    def __init__(self, start_square, end_square, board, is_enpassant=False):
        ''' board is the current chess state
        '''
        self.start_row = start_square[0]
        self.start_col = start_square[1]
        self.end_row = end_square[0]
        self.end_col = end_square[1]
        self.piece_moved = board[self.start_row][self.start_col]
        self.piece_captured = board[self.end_row][self.end_col]
        self.moveID = self.start_row * 1000 + self.start_col * 100 + self.end_row * 10 + self.end_col    # unique identifier
        self.is_pawn_promotion = False
        if (self.piece_moved == 'wp' and self.end_row == 0) or (self.piece_moved == 'bp' and self.end_row == 7):
            self.is_pawn_promotion = True
        self.is_enpassant = is_enpassant
        if self.is_enpassant:
            self.piece_captured == 'wp' if self.piece_moved == 'bp' else 'bp'



    def __eq__(self, other):
        ''' overriding the equals method
        '''
        if isinstance(other, Move):
            return self.moveID == other.moveID
        return False

    
    def get_chess_notation(self):
        ''' get the usual chess notation from positions
        '''
        return self.get_rank_file(self.start_row, self.start_col) + self.get_rank_file(self.end_row, self.end_col)
    

    def get_rank_file(self, r, c):
        return self.cols_to_files[c] + self.row_to_ranks[r]
        

