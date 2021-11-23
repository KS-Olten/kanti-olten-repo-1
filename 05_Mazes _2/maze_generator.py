# Maze generation using recursive backtracker algorithm
# https://en.wikipedia.org/wiki/Maze_generation_algorithm#Recursive_backtracker

from queue import LifoQueue
from PIL import Image
import numpy as np
from random import choice

def print_pretty(listx):
    """returns a grid of a list of lists of numbers

    list of list -> grid"""
    for lists in listx:
        for i in lists:
            print(i,end='\t')
        print()
    print('*************************************************************************************')
    
def generate_example_maze(example = 1):
    
    if example == 1:
        maze_matrix = [
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        ]
        
    if example == 2:
        maze_matrix = [
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 1, 0, 1, 1],
            [1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        ]
    if example == 3:
        maze_matrix = [
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        ]
        
    if example == 4:
        maze_matrix = [
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        ]
        
    if example == 5:
        maze_matrix = [
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        ]
        
    maze_height = len(maze_matrix)
    maze_width = len(maze_matrix[0])

    grid = [[0 for x in range(maze_width)] for x in range(maze_height)]    

    im = Image.new("RGB",(len(maze_matrix[0]), len(maze_matrix)))
    pixels = []

    for x in range(maze_height):
        for y in range(maze_width):
            pixel = (0, 0, 0)
            if maze_matrix[x][y] == 0:
                pixel = (255, 255, 255)

            pixels.append(pixel)

    im.putdata(pixels)
    image = im.resize((400,400),Image.NEAREST)
    return maze_matrix, image, grid

def generate_random_maze(width):
    height = int(width)
    maze = Image.new('RGB', (2*width + 1, 2*height + 1), 'black')
    pixels = maze.load()

    # Create a path on the very top and bottom so that it has an entrance/exit
    pixels[1, 0] = (255, 255, 255)
    pixels[-2, -1] = (255, 255, 255)
    
    stack = LifoQueue()
    cells = np.zeros((width, height))
    cells[0, 0] = 1
    stack.put((0, 0))

    while not stack.empty():
        x, y = stack.get()

        adjacents = []
        if x > 0 and cells[x - 1, y] == 0:
            adjacents.append((x - 1, y))
        if x < width - 1 and cells[x + 1, y] == 0:
            adjacents.append((x + 1, y))
        if y > 0 and cells[x, y - 1] == 0:
            adjacents.append((x, y - 1))
        if y < height - 1 and cells[x, y + 1] == 0:
            adjacents.append((x, y + 1))

        if adjacents:
            stack.put((x, y))

            neighbour = choice(adjacents)
            neighbour_on_img = (neighbour[0]*2 + 1, neighbour[1]*2 + 1)
            current_on_img = (x*2 + 1, y*2 + 1)
            wall_to_remove = (neighbour[0] + x + 1, neighbour[1] + y + 1)

            pixels[neighbour_on_img] = (255, 255, 255)
            pixels[current_on_img] = (255, 255, 255)
            pixels[wall_to_remove] = (255, 255, 255)

            cells[neighbour] = 1
            stack.put(neighbour)
            
    pix = np.array(maze)
    maze_matrix = []

    for row in pix:
        maz_row = []
        for pixel in row:
            if pixel[0] == 255:
                maz_row.append(0)
            else:
                maz_row.append(1)
        maze_matrix.append(maz_row)
        
    image = maze.resize((400,400),Image.NEAREST)
    
    maze_height = len(maze_matrix)
    maze_width = len(maze_matrix[0])

    grid = [[0 for x in range(maze_width)] for x in range(maze_height)]

    return maze_matrix, image, grid

def bfs_solve_maze(maze_matrix, steps=False):
    maze_height = len(maze_matrix)
    maze_width = len(maze_matrix[0])
    solved_maze = [[0 for x in range(maze_width)] for x in range(maze_height)]

    start_x = 0
    start_y = 1

    end_x = len(maze_matrix)-1
    end_y = len(maze_matrix[len(maze_matrix) - 1]) - 2

    solved_maze[start_x][start_y] = 1

    step = 0

    while solved_maze[end_x][end_y] == 0:
        step = step + 1
        for x in range(maze_height):
            for y in range(maze_width):
                if solved_maze[x][y] == step:
                    if (x - 1) >= 0 and maze_matrix[x-1][y] == 0 and solved_maze[x-1][y] == 0:
                        solved_maze[x-1][y] = step + 1
                    if (x + 1) < maze_height and maze_matrix[x + 1][y] == 0 and solved_maze[x + 1][y] == 0:
                        solved_maze[x+1][y] = step + 1
                    if (y - 1) >= 0 and maze_matrix[x][y-1] == 0 and solved_maze[x][y-1] == 0:
                        solved_maze[x][y-1] = step + 1
                    if (y + 1) < maze_width and maze_matrix[x][y + 1] == 0 and solved_maze[x][y + 1] == 0:
                        solved_maze[x][y+1] = step + 1
        if steps:
            print_pretty(solved_maze)
            
    path = []
    x = end_x
    y = end_y

    step = solved_maze[x][y]
    path.append((x, y))

    while step > 1:
        if x - 1 >= 0 and solved_maze[x-1][y] == step - 1:
            x -= 1
            path.append((x, y))
            step -= 1
        if x + 1 < maze_height and solved_maze[x+1][y] == step - 1:
            x += 1
            path.append((x, y))
            step -= 1
        if y - 1 >= 0 and solved_maze[x][y-1] == step - 1:
            y -= 1
            path.append((x, y))
            step -= 1
        if  y + 1 < maze_width and solved_maze[x][y+1] == step - 1:
            y += 1
            path.append((x, y))
            step -= 1
            
    path.pop()
            
    return solved_maze, path

def dfs_solve_maze_get_surrounding(solved_maze, maze_matrix, x, y):
    surrounding = []
    if (x + 1) < len(maze_matrix) and maze_matrix[x + 1][y] == 0 and solved_maze[x + 1][y] == 0:
        surrounding.append((x+1, y))
    if (y + 1) < len(maze_matrix[0]) and maze_matrix[x][y + 1] == 0 and solved_maze[x][y + 1] == 0:
        surrounding.append((x, y+1))
    if (x - 1) >= 0 and maze_matrix[x-1][y] == 0 and solved_maze[x-1][y] == 0:
        surrounding.append((x-1, y))
    if (y - 1) >= 0 and maze_matrix[x][y-1] == 0 and solved_maze[x][y-1] == 0:
        surrounding.append((x, y-1))

    return surrounding
        
def dfs_solve_maze_get_latest_branch(solved_maze, maze_matrix, x, y):
    step = solved_maze[x][y]
    while step > 0:
        step = step -1
        for try_x in [-1, 1]:
            if x + try_x >= 0 and x + try_x < len(solved_maze) and solved_maze[x + try_x][y] == step:
                x = try_x + x
                surrounding = dfs_solve_maze_get_surrounding(solved_maze, maze_matrix, x, y)
                if len(surrounding) > 0:
                    return x, y, step        
        for try_y in [-1, 1]:
            if y + try_y >= 0 and y + try_y < len(solved_maze[0]) and solved_maze[x][y + try_y] == step:
                y = try_y + y
                surrounding = dfs_solve_maze_get_surrounding(solved_maze, maze_matrix, x, y)
                if len(surrounding) > 0:
                    return x, y, step  
                    
    return x, y, step
                    
        

def dfs_solve_maze(maze_matrix, steps=False):
    maze_height = len(maze_matrix)
    maze_width = len(maze_matrix[0])
    solved_maze = [[0 for x in range(maze_width)] for x in range(maze_height)]

    start_x = 0
    start_y = 1

    end_x = len(maze_matrix)-1
    end_y = len(maze_matrix[len(maze_matrix) - 1]) - 2
    

    solved_maze[start_x][start_y] = 1

    step = 1
    x = start_x
    y = start_y

    while solved_maze[end_x][end_y] == 0:
        step = step + 1
        surrounding = dfs_solve_maze_get_surrounding(solved_maze, maze_matrix, x, y)
        if len(surrounding) != 0:
            solved_maze[surrounding[0][0]][surrounding[0][1]] = step
            x = surrounding[0][0]
            y = surrounding[0][1]
        else:
            x, y, step = dfs_solve_maze_get_latest_branch(solved_maze, maze_matrix, x, y)
        if steps:
            print_pretty(solved_maze)
                
    path = []
    x = end_x
    y = end_y

    step = solved_maze[x][y]
    path.append((x, y))

    while step > 1:
        if x - 1 >= 0 and solved_maze[x-1][y] == step - 1:
            x -= 1
            path.append((x, y))
            step -= 1
        if x + 1 < maze_height and solved_maze[x+1][y] == step - 1:
            x += 1
            path.append((x, y))
            step -= 1
        if y - 1 >= 0 and solved_maze[x][y-1] == step - 1:
            y -= 1
            path.append((x, y))
            step -= 1
        if  y + 1 < maze_width and solved_maze[x][y+1] == step - 1:
            y += 1
            path.append((x, y))
            step -= 1
            
    path.pop()
            
    return solved_maze, path
    
def plot_path(maze_matrix, solved_maze, path):
    im = Image.new("RGB",(len(solved_maze[0]), len(solved_maze)))
    pixels = []
    search_count = 0

    for x in range(len(solved_maze)):
        for y in range(len(solved_maze[0])):
            if solved_maze[x][y] != 0:
                search_count += 1
            
            pixel = (0, 0, 0)
            if maze_matrix[x][y] == 0:
                pixel = (255, 255, 255)

            if solved_maze[x][y] > 0:
                pixel = (0, 0, 255)

            if (x, y) in path:
                pixel = (255, 0, 0)

            pixels.append(pixel)

    print('Path Length: {}'.format(len(path)))
    print('Search Stepts: {}'.format(search_count))

    im.putdata(pixels)
    image = im.resize((400,400),Image.NEAREST)
    return image


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("width", nargs="?", type=int, default=32)
    parser.add_argument("height", nargs="?", type=int, default=None)
    parser.add_argument('--output', '-o', nargs='?', type=str, default='generated_maze.png')
    args = parser.parse_args()

    size = (args.width, args.height) if args.height else (args.width, args.width)

    maze = generate_maze(*size)
    maze.save(args.output)