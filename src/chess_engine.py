"""
Storing the information abour the current state of the chess game
Determining the valid moves at the current state. (move log)
"""


from string import whitespace


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
        self.move_log = []
        self.white_king_location = (7, 4)
        self.black_king_location = (0, 4)

    
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
            


    def get_valid_moves(self):
        ''' all moves considering checks 
        '''
        return self.get_all_possible_moves()

    
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
        if self.white_to_move:   # white is playing

            if self.board[r-1][c] == '--':   # 1 square move
                moves.append(Move((r, c), (r-1, c), self.board))
                if r == 6 and self.board[r-2][c] == '--':  # 2 squares move in the beginning
                    moves.append(Move((r, c), (r-2, c), self.board))
            
            if c - 1 >= 0:  # capture to the left
                if self.board[r-1][c-1][0] == 'b':   # ennemy piece to capture 
                    moves.append(Move((r, c), (r-1, c-1), self.board))
            
            if c + 1 <= 7:  # capture to the right
                if self.board[r-1][c+1][0] == 'b':   # ennemy piece to capture 
                    moves.append(Move((r, c), (r-1, c+1), self.board))
            


        else:  # black is playing

            if self.board[r+1][c] == '--':   # 1 square move
                moves.append(Move((r, c), (r+1, c), self.board))
                if r == 1 and self.board[r+2][c] == '--':  # 2 squares move in the beginning
                    moves.append(Move((r, c), (r+2, c), self.board))
            
            if c - 1 >= 0:  # capture to the left (from white perspective...)
                if self.board[r+1][c-1][0] == 'w':   # ennemy piece to capture 
                    moves.append(Move((r, c), (r+1, c-1), self.board))
            
            if c + 1 <= 7:  # capture to the right
                if self.board[r+1][c+1][0] == 'w':   # ennemy piece to capture 
                    moves.append(Move((r, c), (r+1, c+1), self.board))
            



    def get_rook_moves(self, r, c, moves):
        ''' get all the rook possible moves for the rook located at row, col and add them to the list moves
        '''
        if self.white_to_move:
            ennemy = 'b'
        else:
            ennemy = 'w'

        up, down, left, right = r - 1, r + 1, c - 1, c + 1

        while up >= 0 and self.board[up][c] == '--':   # go up when you can
            moves.append(Move((r, c), (up, c), self.board))
            up -= 1
        if up >= 0 and self.board[up][c][0] == ennemy:    # capture
            moves.append(Move((r, c), (up, c), self.board))
        
        while down <= 7 and self.board[down][c] == '--':   # go down when you can
            moves.append(Move((r, c), (down, c), self.board))
            down += 1
        if down <= 7 and self.board[down][c][0] == ennemy:    
            moves.append(Move((r, c), (down, c), self.board))
        
        while left >= 0 and self.board[r][left] == '--':   # go left when you can
            moves.append(Move((r, c), (r, left), self.board))
            left -= 1
        if left >= 0 and self.board[r][left][0] == ennemy:    
            moves.append(Move((r, c), (r, left), self.board))
        
        while right <= 7 and self.board[r][right] == '--':   # go right when you can
            moves.append(Move((r, c), (r, right), self.board))
            right += 1
        if right <= 7 and self.board[r][right][0] == ennemy:   
            moves.append(Move((r, c), (r, right), self.board))
        



    def get_knight_moves(self, r, c, moves):
        ''' get all the knight possible moves for the knight located at row, col and add them to the list moves
        '''
        if self.white_to_move:
            ally = 'w'
        else:
            ally = 'b'

        moves_knight = [(r + 2, c + 1), (r + 2, c - 1), (r - 2, c + 1), (r - 2, c - 1), 
                        (r + 1, c + 2), (r - 1, c + 2), (r + 1, c - 2), (r - 1, c - 2)]
        
        for (x, y) in moves_knight:
            if 0 <= x <= 7 and 0 <= y <= 7 and self.board[x][y][0] != ally:
                moves.append(Move((r, c), (x, y), self.board))



    def get_bishop_moves(self, r, c, moves):
        ''' get all the bishop possible moves for the bishop located at row, col and add them to the list moves
        '''
        if self.white_to_move:
            ennemy = 'b'
        else:
            ennemy = 'w'

        up, left = r - 1, c - 1
        while up >= 0 and c >= 0 and self.board[up][left] == '--':   # go along the top left diagonal
            moves.append(Move((r, c), (up, left), self.board))
            up -= 1
            left -= 1
        if up >= 0 and left >= 0 and self.board[up][left][0] == ennemy:    # capture
            moves.append(Move((r, c), (up, left), self.board))
        
        up, right = r - 1, c + 1
        while up >= 0 and right <= 7 and self.board[up][right] == '--':   # go along the top right diagonal
            moves.append(Move((r, c), (up, right), self.board))
            up -= 1
            right += 1
        if up >= 0 and right <= 7 and self.board[up][right][0] == ennemy:    
            moves.append(Move((r, c), (up, right), self.board))
        
        down, left = r + 1, c - 1
        while down <= 7 and left >= 0 and self.board[down][left] == '--':   # go along the bottom left diagonal
            moves.append(Move((r, c), (down, left), self.board))
            down += 1
            left -= 1
        if down <= 7 and left >= 0 and self.board[down][left][0] == ennemy:    
            moves.append(Move((r, c), (down, left), self.board))
        
        down, right = r + 1, c + 1
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
                moves.append(Move((r, c), (x, y), self.board))


  
        

    
class Move:

    ranks_ro_row = {'1': 7, '2': 6, '3': 5, '4': 4, 
                    '5': 3, '6': 2, '7': 1, '8': 0}
    row_to_ranks = {v: k for k, v in ranks_ro_row.items()}
    files_to_cols = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 
                    'e': 4, 'f': 5, 'g': 6, 'h': 7}
    cols_to_files = {v: k for k, v in files_to_cols.items()}

    def __init__(self, start_square, end_square, board):
        ''' board is the current chess state
        '''
        self.start_row = start_square[0]
        self.start_col = start_square[1]
        self.end_row = end_square[0]
        self.end_col = end_square[1]
        self.piece_moved = board[self.start_row][self.start_col]
        self.piece_captured = board[self.end_row][self.end_col]
        self.moveID = self.start_row * 1000 + self.start_col * 100 + self.end_row * 10 + self.end_col    # unique identifier
    
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
        

