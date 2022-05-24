"""
Main driver file: handle user input and display current GameState
"""

from re import A
from tkinter.tix import MAX

from numpy import square
import pygame
from pygame.locals import *
from src import chess_engine
import os


pygame.init()


WIDTH = HEIGHT = 512
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 15
IMAGES = {}


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
    load_images()
    running = True
    square_selected = () # keep track of the last click of the user (row, col)
    player_clicks = [] # keep track of player clicks (two tuples)

    while running:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
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
                    print(move.get_chess_notation())
                    if move in valid_moves: # the move is valid
                        game_state.make_move(move)
                        move_made = True
                        square_selected = ()   # reset
                        player_clicks = []
                    else:
                        player_clicks = [square_selected]

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z:
                    game_state.undo_move()
                    move_made = True
            
        if move_made:
            valid_moves = game_state.get_valid_moves()
            move_made = False


        draw_game_state(screen, game_state)    
        clock.tick(MAX_FPS)
        pygame.display.flip()


def draw_game_state(screen, game_state):
    
    draw_board(screen)
    draw_pieces(screen, game_state.board)


def draw_board(screen):
    ''' draw the squares
    '''

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





if __name__ == "__main__":
    main()