import pygame
import random
import numpy as np

screen_width = 200
screen_height = 360
num_rows = 18
num_columns = 10

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

pygame.init()

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Minesweeper")
font = pygame.font.Font(None, 45)


class Square:
    square_width = screen_width / num_columns
    square_height = screen_height / num_rows

    def __init__(self, is_bomb,  row, column, num_bombs=0):
        self.is_bomb = is_bomb
        self.row = row
        self.column = column
        self.num_bombs = num_bombs
        self.x = self.column-1 * self.square_width
        self.y = self.row-1 * self.square_height

    def display(self):
        if self.is_bomb:
            pygame.draw.rect(screen, black, [self.x, self.y, self.square_width, self.square_height])
        elif self.num_bombs == 0:
            pygame.draw.rect(screen, white, [self.x, self.y, self.square_width, self.square_height])
        elif self.num_bombs in range(1, 3):
            pygame.draw.rect(screen, green, [self.x, self.y, self.square_width, self.square_height])
        elif self.num_bombs in range(3, 5):
            pygame.draw.rect(screen, blue, [self.x, self.y, self.square_width, self.square_height])
        else:
            pygame.draw.rect(screen, red, [self.x, self.y, self.square_width, self.square_height])


class Grid:
    rows = num_rows
    columns = num_columns
    squares = [[0] * columns] * rows
    print(squares)

    def __init__(self):
        for row in range(0, self.rows):
            for column in range(0, self.columns):
                is_bomb = False
                bomb = random.randrange(0, 5)
                if bomb == 0:
                    is_bomb = True
                self.squares[row][column] = Square(is_bomb, row, column)
        self.find_num_bombs()

    def find_num_bombs(self):
        for row in range(0, self.rows):
            for column in range(0, self.columns):
                if row > 0:
                    if self.squares[row-1][column].is_bomb:
                        self.squares[row][column].num_bombs += 1

                    if column > 0:
                        if self.squares[row-1][column-1].is_bomb:
                            self.squares[row][column].num_bombs += 1

                    if column < (self.columns-1):
                        if self.squares[row-1][column+1].is_bomb:
                            self.squares[row][column].num_bombs += 1

                if column > 0:
                    if self.squares[row][column-1].is_bomb:
                        self.squares[row][column].num_bombs += 1

                    if row < (self.rows-1):
                        if self.squares[row+1][column-1].is_bomb:
                            self.squares[row][column].num_bombs += 1

                if row < (self.rows-1):
                    if self.squares[row+1][column].is_bomb:
                        self.squares[row][column].num_bombs += 1

                    if column < (self.columns-1):
                        if self.squares[row + 1][column+1].is_bomb:
                            self.squares[row][column].num_bombs += 1

                if column < (self.columns-1):
                    if self.squares[row][column+1].is_bomb:
                        self.squares[row][column].num_bombs += 1

                print(self.squares[row][column].is_bomb, self.squares[row][column].num_bombs)

    def display(self):
        for row in range(0, self.rows):
            for column in range(0, self.columns):
                self.squares[row][column].display()


g = Grid()
g.display()
pygame.display.flip()
pygame.time.delay(5000)
pygame.quit()