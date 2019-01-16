import pygame
import pyglet
import random
import math
import numpy as np

screen_width = 200
screen_height = 360
num_rows = 18
num_columns = 10

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 125, 0)
blue = (0, 0, 255)
gray = (185, 185, 185)

pygame.init()

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Minesweeper")
clock = pyglet.clock.Clock()
clock.set_fps_limit(20)
square_img = pygame.image.load("./img/square.png")
square_img = pygame.transform.scale(square_img, (int(screen_width / num_columns), int(screen_height / num_rows)))
mine_img = pygame.image.load("./img/mine.jpeg")
mine_img = pygame.transform.scale(mine_img, (int(screen_width / num_columns), int(screen_height / num_rows)))
flag_img = pygame.image.load("./img/flag.png")
flag_img = pygame.transform.scale(flag_img, (int(screen_width / num_columns), int(screen_height / num_rows)))
font = pygame.font.Font(None, 30)


class Square:
    square_width = screen_width / num_columns
    square_height = screen_height / num_rows
    clicked = False

    def __init__(self, column, row, is_bomb, num_bombs_around=0):
        self.is_bomb = is_bomb
        self.row = row
        self.column = column
        self.num_bombs_around = num_bombs_around
        self.x = self.column * self.square_width
        self.y = self.row * self.square_height

    def display(self):
        if not self.clicked:
            screen.blit(square_img, (self.x, self.y))
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


class Grid:
    rows = num_rows
    columns = num_columns
    num_bombs = 0
    squares = np.empty((columns, rows), dtype=Square)

    def __init__(self):
        for row in range(0, self.rows):
            for column in range(0, self.columns):
                is_bomb = False
                bomb = random.randrange(0, 8)
                if bomb == 0:
                    self.num_bombs += 1
                    is_bomb = True
                self.squares[column][row] = Square(column, row, is_bomb)
        self.find_num_bombs_around()

    def find_num_bombs_around(self):
        for row in range(0, self.rows):
            for column in range(0, self.columns):
                if row > 0:
                    if self.squares[column][row-1].is_bomb:
                        self.squares[column][row].num_bombs_around += 1

                    if column > 0:
                        if self.squares[column-1][row-1].is_bomb:
                            self.squares[column][row].num_bombs_around += 1

                    if column < (self.columns-1):
                        if self.squares[column+1][row-1].is_bomb:
                            self.squares[column][row].num_bombs_around += 1

                if column > 0:
                    if self.squares[column-1][row].is_bomb:
                        self.squares[column][row].num_bombs_around += 1

                    if row < (self.rows-1):
                        if self.squares[column-1][row+1].is_bomb:
                            self.squares[column][row].num_bombs_around += 1

                if row < (self.rows-1):
                    print(row, self.rows)
                    if self.squares[column][row+1].is_bomb:
                        self.squares[column][row].num_bombs_around += 1

                    if column < (self.columns-1):
                        if self.squares[column+1][row + 1].is_bomb:
                            self.squares[column][row].num_bombs_around += 1

                if column < (self.columns-1):
                    if self.squares[column+1][row].is_bomb:
                        self.squares[column][row].num_bombs_around += 1

    def display(self):
        for row in range(0, self.rows):
            for column in range(0, self.columns):
                self.squares[column][row].display()

    def show(self, column, row):
        print(column, row)
        self.squares[column][row].clicked = True
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
        return False

    def show_all_bombs(self):
        for row in range(0, self.rows):
            for column in range(0, self.columns):
                if self.squares[column][row].is_bomb:
                    self.squares[column][row].clicked = True


def play():
    grid = Grid()
    done = False
    bombs_found = 0
    while not done:
        screen.fill(black)
        if bombs_found == grid.num_bombs:
            done = True

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                print(pos)
                x = math.floor(pos[0] / 20)
                y = math.floor(pos[1] / 20)
                print(x, y)
                is_bomb = grid.show(x, y)
                if is_bomb:
                    grid.show_all_bombs()
                    done = True

        clock.tick()
        grid.display()
        pygame.display.flip()


play()
pygame.time.delay(3000)
pygame.quit()
