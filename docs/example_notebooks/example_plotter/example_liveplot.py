import time
import dataanalyzer as da
import numpy as np
import matplotlib.pyplot as plt

from dataanalyzer import Valueclass, Plotter

import matplotlib

matplotlib.use("MACOSX")

propTrue = 0.3
arrayDims = (200, 200)

randomArray = np.random.choice(
    a=[True, False], size=arrayDims, p=[propTrue, 1 - propTrue]
)
randomValueclass = Valueclass(randomArray, name="randomValueclass")

x = Valueclass(np.linspace(0, 10, arrayDims[0]), name="x")
y = Valueclass(np.linspace(0, 10, arrayDims[1]), name="y")


def check_neighbours(matrix, x, y):
    submatrix = np.copy(matrix[x - 1 : x + 2, y - 1 : y + 2])
    submatrix[1, 1] = False
    nTrue = np.sum(submatrix)

    if nTrue < 2 or nTrue > 3:
        return False
    elif nTrue == 3:
        return True
    return matrix[x, y]


def update_matrix(matrix):
    newMatrix = np.zeros_like(matrix)
    for x in range(1, matrix.shape[0] - 1):
        for y in range(1, matrix.shape[1] - 1):
            newMatrix[x, y] = check_neighbours(matrix, x, y)
    return newMatrix


def update_valueclass(valueclass):
    valueclass.value = update_matrix(valueclass.value)
    return valueclass


def get_scatter_points(matrix, x, y):
    xy_points = np.array((np.meshgrid(x.value, y.value))).T.reshape(-1, 2)
    x_points = xy_points[matrix.value.reshape(-1), 0]
    y_points = xy_points[matrix.value.reshape(-1), 1]
    return x_points, y_points


def move_snake(matrix, direction="UP", prob: int = 1, fixed: bool = False):
    x, y = np.where(matrix.value)
    x, y = x[0], y[0]
    matrix.value[x, y] = False

    # get opposite direction
    if direction == "UP":
        opposite = "DOWN"
    elif direction == "DOWN":
        opposite = "UP"
    elif direction == "LEFT":
        opposite = "RIGHT"
    else:
        opposite = "LEFT"

    # pick the direction by random, but not the opposite of the old direction
    directions = ["UP", "DOWN", "LEFT", "RIGHT"]

    directions.remove(opposite)
    directions += [direction] * prob

    if not fixed:
        direction = np.random.choice(directions)

    if direction == "UP":
        x -= 1
    elif direction == "DOWN":
        x += 1
    elif direction == "LEFT":
        y -= 1
    elif direction == "RIGHT":
        y += 1

    matrix.value[x, y] = True
    return matrix, direction


####################################################################################################
#                 Live Plot with Heatmap                                                           #
####################################################################################################
if False:
    fig = plt.figure()
    for _ in range(1000):
        plot = Plotter(fig=fig)
        randomValueclass = update_valueclass(randomValueclass)
        plot.heatmap(
            x,
            y,
            Z=randomValueclass,
            title="Conway's Game of Life",
            add_colorbar=False,
            cmap="binary",
        )
        fig = plot.show(return_fig=True)
        plt.pause(0.2)

####################################################################################################
#                 Live Plot with Scatter                                                           #
####################################################################################################
if False:
    fig = plt.figure()
    xPoints_list, yPoints_list = [], []

    for _ in range(1000):
        plot = Plotter(fig=fig)
        randomValueclass = update_valueclass(randomValueclass)
        xPoints, yPoints = get_scatter_points(randomValueclass, x, y)

        for i, [xPoints_old, yPoints_old] in enumerate(
            zip(xPoints_list[::-1], yPoints_list[::-1])
        ):
            if not (alpha := max(0, 1 - 0.2 * (i + 1))):
                break
            plot.scatter(xPoints_old, yPoints_old, alpha=alpha, c="lightgrey", s=5)

        plot.scatter(xPoints, yPoints, title="Conway's Game of Life", s=3)

        plot.ax.set_xlim(0, 10)
        plot.ax.set_ylim(0, 10)
        fig = plot.show(return_fig=True)
        plt.pause(0.2)

        xPoints_list.append(xPoints)
        yPoints_list.append(yPoints)

####################################################################################################
#                 Live Plot (Snake)                                                                #
####################################################################################################
if True:
    snakeTailx, snakeTaily = [], []

    # Create array with one True value in the middle
    snakeDims = (50, 50)
    snakeArray = np.zeros(snakeDims, dtype=bool)
    snakeArray[snakeDims[0] // 2, snakeDims[1] // 2] = True
    snakeValueclass = Valueclass(snakeArray, name="snakeValueclass")

    # Make snake food by randomly placing True values
    numberFood = 100
    foodArray = np.zeros(snakeDims, dtype=bool)
    for _ in range(numberFood):
        x, y = np.random.randint(0, snakeDims[0]), np.random.randint(0, snakeDims[1])
        foodArray[x, y] = True
    foodValueclass = Valueclass(foodArray, name="foodValueclass")

    xSnake = Valueclass(np.linspace(0, 10, snakeDims[0]), name="xSnake")
    ySnake = Valueclass(np.linspace(0, 10, snakeDims[1]), name="ySnake")
    pointSize = snakeArray.size / 25

    fig = plt.figure()
    xTrace_list, yTrace_list = [], []
    directions = []
    direction = "UP"
    for _ in range(1000):
        # update snake and food
        plot = Plotter(fig=fig)
        snakeValueclass, direction = move_snake(snakeValueclass, direction=direction)
        xHead, yHead = get_scatter_points(snakeValueclass, xSnake, ySnake)
        directions.append(direction)

        # plot old snake
        for i, [xHead_old, yHead_old] in enumerate(
            zip(xTrace_list[::-1], yTrace_list[::-1])
        ):
            if not (alpha := max(0, 1 - 0.2 * (i + 1))):
                break
            plot.scatter(
                xHead_old,
                yHead_old,
                alpha=alpha,
                c="lightgrey",
                s=pointSize,
                marker="s",
            )

        # plot food
        xFood, yFood = get_scatter_points(foodValueclass, xSnake, ySnake)
        plot.scatter(xFood, yFood, c="red", s=pointSize, marker="o")

        # plot snake tail
        if len(snakeTailx) > 1:
            plot.scatter(snakeTailx, snakeTaily, c="lightblue", s=pointSize, marker="s")

        # update snake tail
        x, y = np.where(snakeValueclass.value)
        x, y = x[0], y[0]

        if foodValueclass.value[x, y]:
            foodValueclass.value[x, y] = False
            snakeTailx.append(x)
            snakeTaily.append(y)

        # move snake tail
        if len(snakeTailx) > 1:
            for i in range(len(snakeTailx) - 1):
                # make matrix of snake tail i
                snakeTailArray = np.zeros(snakeDims, dtype=bool)
                snakeTailArray[snakeTailx[i], snakeTaily[i]] = True
                snakeTailValueclass = Valueclass(
                    snakeTailArray, name="snakeTailValueclass"
                )
                # move snake tail i
                snakeTailValueclass, direction = move_snake(
                    snakeTailValueclass, direction=directions[i], fixed=True
                )
                xTaili, yTaili = get_scatter_points(snakeTailValueclass, xSnake, ySnake)

        # plot snake head
        plot.scatter(xHead, yHead, title="Snake", s=pointSize, marker="s")

        # plot settings
        plot.ax.set_xlim(0, 10)
        plot.ax.set_ylim(0, 10)
        fig = plot.show(return_fig=True)
        plt.pause(0.2)

        xTrace_list.append(xHead)
        yTrace_list.append(yHead)
