import random
import math
import numpy as np


class RacetrackGenerator:

    def __init__(self):
        self.racetrack = []
        self.rows = 0
        self.columns = 0
        self.start_cols = 0
        self.finish_coords = {}

    def generate_racetrack(self, rows, cols):
        self.rows = rows
        self.columns = cols
        # Initialize racetrack with 'X' as out of bounds
        self.racetrack = [['X'] * cols for i in range(rows)]

        # Chose random width for the start line also choose where to start creating it (left, middle, right)
        s_genesis = 'middle' #random.choice(['left', 'middle', 'right'])
        s_width = random.randint(math.ceil(0.35 * cols), cols)
        start_line_idx = []
        if s_genesis == 'left':
            start_line_idx = [0, s_width-1]
            for i in range(s_width):
                self.racetrack[rows - 1][i] = 'S'
                self.racetrack[rows - 2][i] = '*'
        elif s_genesis == 'middle':
            start_line_idx = [cols // 2 - s_width // 2, cols // 2 + s_width // 2 - 1]
            j = 1
            for i in range(cols // 2, cols // 2 + s_width // 2, 1):
                self.racetrack[rows - 1][i] = 'S'
                self.racetrack[rows - 1][i-j] = 'S'
                self.racetrack[rows - 2][i] = '*'
                self.racetrack[rows - 2][i - j] = '*'
                j += 2
        elif s_genesis == 'right':
            start_line_idx = [cols-s_width, cols-1]
            for i in range(cols-1, cols-1-s_width, -1):
                self.racetrack[rows-1][i] = 'S'
                self.racetrack[rows - 2][i] = '*'

        # Choose random length for finish line
        f_length = random.randint(math.ceil(0.1 * rows), math.ceil(0.2 * rows))
        for i in range(f_length):
            self.racetrack[i][cols - 1] = 'F'

        # Idea: start from start line -> next row -> check if finish line is in this row -> no: find random width and
        # position where to start putting '*', yes: start printing '*' towards the finish line ...
        path_start = random.randint(start_line_idx[0], start_line_idx[1])
        path_span_prob = random.uniform(0.4, 0.6)
        path_span = random.randint(math.ceil(0.35 * cols), math.ceil(path_span_prob * cols))
        for current_row in range(rows-3, -1, -1):
            if 'F' in self.racetrack[current_row]:
                path_span = math.ceil(0.1 * cols)
            j = 1
            end_path = 0
            start_path = 0
            for i in range(path_start, path_start+path_span//2, 1):
                if i < cols:
                    self.racetrack[current_row][i] = '*' if self.racetrack[current_row][i] != 'F' else 'F'
                    end_path = i
                if i-j >= 0:
                    self.racetrack[current_row][i-j] = '*'
                    start_path = i-j
                j += 2
            path_start = random.randint(start_path, end_path)

        # Check if Finish line is connected
        for i in range(f_length):
            for j in range(cols-2, 0, -1):
                if self.racetrack[i][j] == 'X':
                    self.racetrack[i][j] = '*'
                else:
                    break

        # Return racetrack
        return self.racetrack

    def racetrack1_from_book(self):
        self.racetrack = np.full((32, 18), '*', dtype=str)
        self.racetrack[31, 4:10] = 'S'
        self.racetrack[0:6, 17] = 'F'
        self.racetrack[0, 1:4] = 'X'
        self.racetrack[1:3, 1:3] = 'X'
        self.racetrack[3][1] = 'X'
        self.racetrack[6, 11:] = 'X'
        self.racetrack[7:, 10:] = 'X'
        self.racetrack[14:22, 1] = 'X'
        self.racetrack[22:29, 1:3] = 'X'
        self.racetrack[29:, 1:4] = 'X'
        self.racetrack[0:, 0] = 'X'

        self.rows = 32
        self.columns = 18

        return self.racetrack

    def racetrack2_from_book(self):
        self.racetrack = np.full((30, 33), '*', dtype=np.str)
        self.racetrack[0:, 0] = 'X'
        self.racetrack[0:27, 1] = 'X'
        self.racetrack[0:26, 2] = 'X'
        self.racetrack[0:25, 3] = 'X'
        self.racetrack[0:24, 4] = 'X'
        self.racetrack[0:23, 5] = 'X'
        self.racetrack[0:22, 6] = 'X'
        self.racetrack[0:21, 7] = 'X'
        self.racetrack[0:20, 8] = 'X'
        self.racetrack[0:19, 9] = 'X'
        self.racetrack[0:18, 10] = 'X'
        self.racetrack[0:17, 11] = 'X'
        self.racetrack[0:3, 12] = 'X'
        self.racetrack[7:16, 12] = 'X'
        self.racetrack[0:2, 13] = 'X'
        self.racetrack[8:15, 13] = 'X'
        self.racetrack[0:1, 14:16] = 'X'
        self.racetrack[9:14, 14] = 'X'
        self.racetrack[0:9, 32] = 'F'
        self.racetrack[9:, 31:] = 'X'
        self.racetrack[10:, 28:31] = 'X'
        self.racetrack[11:, 27] = 'X'
        self.racetrack[12:, 25:27] = 'X'
        self.racetrack[13:, 24] = 'X'
        self.racetrack[29, 1:24] = 'S'

        self.rows = 30
        self.columns = 33

        return self.racetrack

    def print_racetrack(self):
        for row in self.racetrack.tolist():
            print(''.join(row))

    def return_dimensions(self):
        return self.rows, self.columns
