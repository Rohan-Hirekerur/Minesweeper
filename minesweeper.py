import pygame
import pyglet
import random
import math
import numpy as np

# Initialize numer of rows and columns in the game
num_rows = 12
num_columns = 18

# Scaling factor to display grid
multiplier = 30
screen_width = num_columns * multiplier
screen_height = num_rows * multiplier

# Basic color RGB values required in the game
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 125, 0)
blue = (0, 0, 255)
gray = (185, 185, 185)

# Initialize Pygame
pygame.init()

# Set window parameters (size, caption, etc)
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Minesweeper")

# Load background images for squares, flags and mines
# Transform them according to screen/window size
square_img = pygame.image.load("./img/square.png")
square_img = pygame.transform.scale(square_img, (int(screen_width / num_columns), int(screen_height / num_rows)))
mine_img = pygame.image.load("./img/mine.jpeg")
mine_img = pygame.transform.scale(mine_img, (int(screen_width / num_columns), int(screen_height / num_rows)))
flag_img = pygame.image.load("./img/flag.png")
flag_img = pygame.transform.scale(flag_img, (int(screen_width / num_columns), int(screen_height / num_rows)))
font = pygame.font.Font(None, int(multiplier*1.5))

# Clock
clock = pyglet.clock.Clock()
clock.set_fps_limit(20)


# Class square is for each small square in the grid
class Square:
    square_width = screen_width / num_columns
    square_height = screen_height / num_rows
    clicked = False
    flagged = False

    def __init__(self, column, row, is_bomb, num_bombs_around=0):
        self.is_bomb = is_bomb
        self.row = row
        self.column = column
        self.num_bombs_around = num_bombs_around
        self.x = self.column * self.square_width
        self.y = self.row * self.square_height

    # Display function shows the number of mines around that square if the square is clicked
    # Else, it just shows the empty square
    def display(self):
        if not self.clicked:
            if not self.flagged:
                screen.blit(square_img, (self.x, self.y))
            else:
                screen.blit(flag_img, (self.x, self.y))
        if self.clicked:
            pygame.draw.rect(screen, gray, [self.x, self.y, self.square_width, self.square_height])
            if self.is_bomb:
                screen.blit(mine_img, (self.x, self.y))
            elif self.num_bombs_around == 0:
                text = font.render("", 1, black)
                text_pos = (self.x, self.y)
                screen.blit(text, text_pos)
            elif self.num_bombs_around == 1:
                text = font.render("1", 1, green)
                text_pos = (self.x, self.y)
                screen.blit(text, text_pos)
            elif self.num_bombs_around == 2:
                text = font.render("2", 1, blue)
                text_pos = (self.x, self.y)
                screen.blit(text, text_pos)
            elif self.num_bombs_around == 3:
                text = font.render("3", 1, red)
                text_pos = (self.x, self.y)
                screen.blit(text, text_pos)
            else:
                t = str(self.num_bombs_around)
                text = font.render(t, 1, black)
                text_pos = (self.x, self.y)
                screen.blit(text, text_pos)


# This is the main class for the game grid.
class Grid:
    rows = num_rows
    columns = num_columns
    num_bombs = 0
    # Empty array to store our squares
    squares = np.empty((columns, rows), dtype=Square)

    # This constructor initializes the squares of grid individually
    # While creating each square, it is assigned a bomb with the probability given in randrange
    # eg: 1/6 for randrange(0, 6)
    def __init__(self):
        for row in range(0, self.rows):
            for column in range(0, self.columns):
                is_bomb = False
                bomb = random.randrange(0, 6)
                if bomb == 0:
                    self.num_bombs += 1
                    is_bomb = True
                self.squares[column][row] = Square(column, row, is_bomb)
        self.find_num_bombs_around()

    # For each square, its neighbouring squares are checked for mines/bombs
    # If present, number of bombs is incremented
    # Checking at position "X" as later mentioned in comments
    def find_num_bombs_around(self):
        for row in range(0, self.rows):
            for column in range(0, self.columns):
                if row > 0:                                                         # _ X _
                    if self.squares[column][row-1].is_bomb:                         # _   _
                        self.squares[column][row].num_bombs_around += 1             # _ _ _

                    if column > 0:                                                  # X _ _
                        if self.squares[column-1][row-1].is_bomb:                   # _   _
                            self.squares[column][row].num_bombs_around += 1         # _ _ _

                    if column < (self.columns-1):                                   # _ _ X
                        if self.squares[column+1][row-1].is_bomb:                   # _   _
                            self.squares[column][row].num_bombs_around += 1         # _ _ _

                if column > 0:                                                      # _ _ _
                    if self.squares[column-1][row].is_bomb:                         # X   _
                        self.squares[column][row].num_bombs_around += 1             # _ _ _

                    if row < (self.rows-1):                                         # _ _ _
                        if self.squares[column-1][row+1].is_bomb:                   # _   _
                            self.squares[column][row].num_bombs_around += 1         # X _ _

                if row < (self.rows-1):                                             # _ _ _
                    if self.squares[column][row+1].is_bomb:                         # _   _
                        self.squares[column][row].num_bombs_around += 1             # _ X _

                    if column < (self.columns-1):                                   # _ _ _
                        if self.squares[column+1][row + 1].is_bomb:                 # _   _
                            self.squares[column][row].num_bombs_around += 1         # _ _ X

                if column < (self.columns-1):                                       # _ _ _
                    if self.squares[column+1][row].is_bomb:                         # _   X
                        self.squares[column][row].num_bombs_around += 1             # _ _ _

    # Display complete grid
    def display(self):
        for row in range(0, self.rows):
            for column in range(0, self.columns):
                self.squares[column][row].display()

    # This function shows the clicked square
    # If the clicked square is blank, all its surrounding squares are also shown
    # The function is recursively called to check if the surrounding squares contain empty square
    def show(self, column, row):
        print(column, row)
        self.squares[column][row].clicked = True
        self.squares[column][row].flagged = False
        if self.squares[column][row].is_bomb:
            return True
        if self.squares[column][row].num_bombs_around == 0:
            print("In")
            if column > 0:
                if not self.squares[column-1, row].clicked:
                    self.show(column-1, row)
            if column < self.columns-1:
                if not self.squares[column+1, row].clicked:
                    self.show(column+1, row)
            if row > 0:
                if not self.squares[column, row-1].clicked:
                    self.show(column, row-1)
            if row < self.rows-1:
                if not self.squares[column, row+1].clicked:
                    self.show(column, row+1)
            if column > 0 and row > 0:
                if not self.squares[column-1, row-1].clicked:
                    self.show(column-1, row-1)
                self.squares[column - 1][row - 1].clicked = True
            if column < self.columns-1 and row < self.rows-1:
                if not self.squares[column+1, row+1].clicked:
                    self.show(column+1, row+1)
                self.squares[column + 1][row + 1].clicked = True
            if column > 0 and row < self.rows-1:
                if not self.squares[column-1, row+1].clicked:
                    self.show(column-1, row+1)
                self.squares[column - 1][row + 1].clicked = True
            if column < self.columns-1 and row > 0:
                if not self.squares[column+1, row-1].clicked:
                    self.show(column+1, row-1)
                self.squares[column + 1][row - 1].clicked = True
        return False

    # Show all mines/bombs when game over
    def show_all_bombs(self):
        for row in range(0, self.rows):
            for column in range(0, self.columns):
                if self.squares[column][row].is_bomb:
                    self.squares[column][row].clicked = True

    # Flag a mine
    def flag(self, column, row, bombs_found):
        if self.squares[column][row].flagged:
            self.squares[column][row].flagged = False
            bombs_found -= 1
        else:
            self.squares[column][row].flagged = True
            bombs_found += 1
        return bombs_found

    # Validate the game completion
    def validate(self):
        for row in range(0, self.rows):
            for column in range(0, self.columns):
                if not self.squares[column][row].clicked and not self.squares[column][row].flagged:
                    return False
        return True


# Here we actually play the game
def play():
    grid = Grid()
    done = False
    bombs_found = 0
    while not done:
        screen.fill(black)
        if bombs_found == grid.num_bombs:
            done = grid.validate()

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                print(pos)
                x = math.floor(pos[0] / multiplier)
                y = math.floor(pos[1] / multiplier)
                print(x, y)
                if event.button == 1:               # 1 --> Left click
                    is_bomb = grid.show(x, y)
                    if is_bomb:
                        grid.show_all_bombs()
                        done = True
                if event.button == 3:               # 3 --> Right click
                    bombs_found = grid.flag(x, y, bombs_found)

        clock.tick()
        grid.display()
        pygame.display.flip()


play()
pygame.time.delay(3000)
pygame.quit()
