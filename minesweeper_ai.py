from collections import deque
import pygame
import pyglet
import random
import math
import numpy as np
import cnn
import tensorflow as tf

state_size = [200, 200, 3]
action_size = 100
learning_rate = 0.0002      # Alpha (aka learning rate)

# TRAINING HYPER PARAMETERS
total_episodes = 1500        # Total episodes for training
max_steps = 100              # Max possible steps in an episode
batch_size = 100

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Q learning hyper parameters
gamma = 0.95               # Discounting rate

# MEMORY HYPER PARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

# MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = False
testing = True


possible_actions = []
for i in range(0, 100):
    possible_actions.append(i)

# Initialize number of rows and columns in the game
num_rows = 10
num_columns = 10

# Scaling factor to display grid
multiplier = 20
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
    num_clicked = 0
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
        if not self.squares[column][row].clicked:
            self.num_clicked += 1
        self.squares[column][row].clicked = True
        self.squares[column][row].flagged = False
        if self.squares[column][row].is_bomb:
            return True
        if self.squares[column][row].num_bombs_around == 0:
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
            if column < self.columns-1 and row < self.rows-1:
                if not self.squares[column+1, row+1].clicked:
                    self.show(column+1, row+1)
            if column > 0 and row < self.rows-1:
                if not self.squares[column-1, row+1].clicked:
                    self.show(column-1, row+1)
            if column < self.columns-1 and row > 0:
                if not self.squares[column+1, row-1].clicked:
                    self.show(column+1, row-1)
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


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    # EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    # First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if explore_probability > exp_exp_tradeoff:
        # Make a random action (exploration)
        action = random.choice(possible_actions)

    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(nn.output, feed_dict={nn.inputs: state.reshape((1, *state.shape))})

        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]

    return action, explore_probability


tf.reset_default_graph()

# Instantiate the nn
nn = cnn.Cnn(state_size, action_size, learning_rate)


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]


memory = Memory(max_size=memory_size)

# Render the environment

"""
for i in range(total_episodes):
    # If it's the first step
    state = pygame.surfarray.array3d(screen)

    grid = Grid()
    done = False
    bombs_found = 0
    while not done:
        reward = 0
        screen.fill(black)

        action = random.choice(possible_actions)

        if bombs_found == grid.num_bombs:
            done = grid.validate()

        events = pygame.event.get()
        x = math.floor(action/10)
        y = action%10
        is_bomb = grid.show(x, y)
        if is_bomb:
            grid.show_all_bombs()
            reward = -101
            done = True
            next_state = np.zeros(state.shape)
            memory.add((state, action, reward, next_state, done))
        else:
            reward = 1
        if grid.num_clicked == (grid.rows * grid.columns) - grid.num_bombs:
            reward = 101
            done = True

        grid.display()
        pygame.display.flip()
        next_state = pygame.surfarray.array3d(screen)
        state = next_state
        memory.add((state, action, reward, next_state, done))
"""
saver = tf.train.Saver()

if training:
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0

        for episode in range(total_episodes):
            # Set step to 0
            step = 0

            # Initialize the rewards of the episode
            episode_rewards = []

            # Make a new episode and observe the first state
            state = pygame.surfarray.array3d(screen)
            grid = Grid()
            done = False
            bombs_found = 0
            actions = []

            while step < max_steps:
                step += 1
                # Increase decay_step
                decay_step += 50

                reward = 0
                screen.fill(black)

                while True:
                    action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step,
                                                                 state, possible_actions)
                    if action not in actions:
                        break
                    # reward -= 1
                actions.append(action)

                if bombs_found == grid.num_bombs:
                    done = grid.validate()

                events = pygame.event.get()
                x = math.floor(action / 10)
                y = action % 10
                print (x, y)
                is_bomb = grid.show(x, y)
                if is_bomb:
                    grid.show_all_bombs()
                    reward -= 101
                    step = max_steps
                    next_state = np.zeros(state.shape, dtype=np.int)
                    memory.add((state, action, reward, next_state, done))
                else:
                    reward += 2
                if grid.num_clicked == (grid.rows * grid.columns) - grid.num_bombs:
                    reward -= 101
                    step = max_steps

                episode_rewards.append(reward)
                grid.display()
                pygame.display.flip()
                next_state = pygame.surfarray.array3d(screen)
                memory.add((state, action, reward, next_state, done))
                state = next_state

                # LEARNING PART
                # Obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []

                # Get Q values for next_state
                Qs_next_state = sess.run(nn.output, feed_dict={nn.inputs: next_states_mb})

                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])

                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)

                targets_mb = np.array([each for each in target_Qs_batch])

                loss, _ = sess.run([nn.loss, nn.optimizer],
                                   feed_dict={nn.inputs: states_mb,
                                              nn.sample_op: targets_mb,
                                              nn.actions: actions_mb})

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./model.ckpt")
                print("Model Saved")

# Here we actually play the game
# def play():
#     grid = Grid()
#     done = False
#     bombs_found = 0
#     while not done:
#         screen.fill(black)
#         if bombs_found == grid.num_bombs:
#             done = grid.validate()
#
#         events = pygame.event.get()
#         for event in events:
#             if event.type == pygame.QUIT or grid.num_clicked == (grid.rows*grid.columns)-grid.num_bombs:
#                 done = True
#             if event.type == pygame.MOUSEBUTTONUP:
#                 pos = pygame.mouse.get_pos()
#                 x = math.floor(pos[0] / multiplier)
#                 y = math.floor(pos[1] / multiplier)
#                 if event.button == 1:               # 1 --> Left click
#                     is_bomb = grid.show(x, y)
#                     if is_bomb:
#                         grid.show_all_bombs()
#                         done = True
#                 if event.button == 3:               # 3 --> Right click
#                     bombs_found = grid.flag(x, y, bombs_found)
#
#         clock.tick()
#         grid.display()
#         pygame.display.flip()
#
#
# play()

if testing:
    with tf.Session() as sess:
        # Initialize the variables
        saver.restore(sess, "./model.ckpt")
        sess.run(tf.global_variables_initializer())

        for episode in range(1):
            state = pygame.surfarray.array3d(screen)
            grid = Grid()
            done = False
            bombs_found = 0
            actions = []

            while not done:
                screen.fill(black)
                Qs = sess.run(nn.output, feed_dict={nn.inputs: state.reshape((1, *state.shape))})

                while True:
                    choice = np.argmax(Qs)
                    action = possible_actions[int(choice)]
                    Qs[0][choice] = -1000
                    if action not in actions:
                        break
                actions.append(action)

                if bombs_found == grid.num_bombs:
                    done = grid.validate()

                events = pygame.event.get()
                x = math.floor(action / 10)
                y = action % 10
                print(x, y)
                is_bomb = grid.show(x, y)
                if is_bomb:
                    grid.show_all_bombs()
                    done = True
                    next_state = np.zeros(state.shape, dtype=np.int)

                if grid.num_clicked == (grid.rows * grid.columns) - grid.num_bombs:
                    done = True

                pygame.time.delay(500)
                grid.display()
                pygame.display.flip()
                next_state = pygame.surfarray.array3d(screen)
                state = next_state

pygame.time.delay(3000)
pygame.quit()
