"""
Main driver file: handle user input and display current GameState
"""

from re import A
# from tkinter.tix import MAX

from numpy import square
import pygame
from pygame.locals import *
from src import chess_engine
from src.minmax_agent import get_best_move as get_best_move_minmax
from src.mcts_agent import get_best_move as get_best_move_mcts
from src.ql_agent import get_best_move as get_best_move_ql, DQNAgent
import os
import torch

pygame.init()


WIDTH = HEIGHT = 512
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 30
IMAGES = {}


agent = DQNAgent()
agent.policy_net.load_state_dict(torch.load("dqn_policy.pth"))
agent.policy_net.eval()

def load_images():
    ''' fill IMAGES to easily access an image with IMAGES['wp'] 
    '''

    pieces = os.listdir('images')

    for piece in pieces: 
        IMAGES[piece[:-4]] = pygame.transform.scale(pygame.image.load('images/' + piece), (SQ_SIZE, SQ_SIZE))

    return IMAGES


def main():
    ''' main function of this file
    '''
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    screen.fill(pygame.Color('white'))
    game_state = chess_engine.GameState()
    valid_moves = game_state.get_valid_moves()
    move_made = False # when a move is made
    animate = True
    load_images()
    running = True
    square_selected = () # keep track of the last click of the user (row, col)
    player_clicks = [] # keep track of player clicks (two tuples)
    player_one = True # if a human is playing white
    player_two = False # if a human is playing black
    game_over = False
    move_undone = False

    while running:
        # for row in game_state.board:
        #     print(" ".join(row))
        # print()

        human_turn = (game_state.white_to_move and player_one) or (not game_state.white_to_move and player_two)
        if not human_turn and not game_over:
            #ai_move = get_best_move_ql(game_state, agent)
            ai_move = get_best_move_mcts(game_state)
            # ai_move = get_best_move_minmax(game_state, depth=1)
            game_state.make_move(ai_move)
            move_made = True

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if not game_over:
                    location = pygame.mouse.get_pos()   # x, y position of the mouse 
                    col = location[0] // SQ_SIZE
                    row = location[1] // SQ_SIZE
                    if square_selected == (row, col): # click the same square twice
                        square_selected = ()
                        player_clicks = []
                    else:
                        square_selected = (row, col)
                        player_clicks.append((row, col))

                    if len(player_clicks) == 2: # it is the second click of the player
                        move = chess_engine.Move(player_clicks[0], player_clicks[1], game_state.board)
                        for i in range(len(valid_moves)):   
                            if move == valid_moves[i]: # the move is valid
                                game_state.make_move(valid_moves[i])  # valid_moves[i] and not make_move otherwise some bugs
                                move_made = True
                                square_selected = ()   # reset
                                player_clicks = []
                        if not move_made:
                            player_clicks = [square_selected]

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z:
                    game_state.undo_move()
                    move_made = True
                    animate = False
                    game_over = False
                    move_undone = True
                if event.key == pygame.K_r:
                    game_state = chess_engine.GameState()
                    valid_moves = game_state.get_valid_moves()
                    square_selected = ()
                    player_clicks = []
                    move_made = False
                    animate = False
                    move_undone = True
                
            
            
        if move_made:
            print(move.get_chess_notation())
            if animate:
                animate_move(game_state.move_log[-1], screen, game_state.board, clock)
            valid_moves = game_state.get_valid_moves()
            move_made = False
            animate = False
            move_undone = False

        draw_game_state(screen, game_state, valid_moves, square_selected)

        if game_state.is_checkmate():
            game_over = True
            if game_state.white_to_move:
                draw_end_game_text(screen, "Black wins by checkmate")
            else:
                draw_end_game_text(screen, "White wins by checkmate")
        elif game_state.is_stalemate():
            game_over = True
            draw_end_game_text(screen, "Stalemate")


        clock.tick(MAX_FPS)
        pygame.display.flip()


def draw_game_state(screen, game_state, valid_moves, square_selected):
    
    draw_board(screen)
    highlight_squares(screen, game_state, valid_moves, square_selected)
    draw_pieces(screen, game_state.board)


def draw_board(screen):
    ''' draw the squares
    '''
    global colors
    colors = [pygame.Color('white'), pygame.Color('gray')]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            pygame.draw.rect(screen, colors[(r+c)%2], pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
    



def draw_pieces(screen, board):
    ''' draw the pieces using the current GameState.board
    '''
    
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]
            if piece != '--':
                screen.blit(IMAGES[piece], pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))


def highlight_squares(screen, game_state, valid_moves, square_selected):
    """
    Highlight square selected and moves for piece selected.
    """
    if (len(game_state.move_log)) > 0:
        last_move = game_state.move_log[-1]
        s = pygame.Surface((SQ_SIZE, SQ_SIZE))
        s.set_alpha(100)
        s.fill(pygame.Color('green'))
        screen.blit(s, (last_move.end_col * SQ_SIZE, last_move.end_row * SQ_SIZE))
    if square_selected != ():
        row, col = square_selected
        if game_state.board[row][col][0] == (
                'w' if game_state.white_to_move else 'b'):  # square_selected is a piece that can be moved
            # highlight selected square
            s = pygame.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(100)  # transparency value 0 -> transparent, 255 -> opaque
            s.fill(pygame.Color('blue'))
            screen.blit(s, (col * SQ_SIZE, row * SQ_SIZE))
            # highlight moves from that square
            s.fill(pygame.Color('yellow'))
            for move in valid_moves:
                if move.start_row == row and move.start_col == col:
                    screen.blit(s, (move.end_col * SQ_SIZE, move.end_row * SQ_SIZE))



def draw_end_game_text(screen, text):
    font = pygame.font.SysFont("Helvetica", 32, True, False)
    text_object = font.render(text, False, pygame.Color("gray"))
    text_location = pygame.Rect(0, 0, WIDTH, HEIGHT).move(WIDTH / 2 - text_object.get_width() / 2,
                                                                 HEIGHT / 2 - text_object.get_height() / 2)
    screen.blit(text_object, text_location)
    text_object = font.render(text, False, pygame.Color('black'))
    screen.blit(text_object, text_location.move(2, 2))


def animate_move(move, screen, board, clock):
    """
    Animating a move
    """
    global colors
    d_row = move.end_row - move.start_row
    d_col = move.end_col - move.start_col
    frames_per_square = 5  # frames to move one square
    frame_count = (abs(d_row) + abs(d_col)) * frames_per_square
    for frame in range(frame_count + 1):
        row, col = (move.start_row + d_row * frame / frame_count, move.start_col + d_col * frame / frame_count)
        draw_board(screen)
        draw_pieces(screen, board)
        # erase the piece moved from its ending square
        color = colors[(move.end_row + move.end_col) % 2]
        end_square = pygame.Rect(move.end_col * SQ_SIZE, move.end_row * SQ_SIZE, SQ_SIZE, SQ_SIZE)
        pygame.draw.rect(screen, color, end_square)
        # draw captured piece onto rectangle
        if move.piece_captured != '--':
            if move.is_enpassant:
                enpassant_row = move.end_row + 1 if move.piece_captured[0] == 'b' else move.end_row - 1
                end_square =pygame.Rect(move.end_col * SQ_SIZE, enpassant_row * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            screen.blit(IMAGES[move.piece_captured], end_square)
        # draw moving piece
        screen.blit(IMAGES[move.piece_moved], pygame.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))
        pygame.display.flip()
        clock.tick(60)




if __name__ == "__main__":
    main()