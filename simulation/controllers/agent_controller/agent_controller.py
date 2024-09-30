import base64
import io
import json
import multiprocessing
import threading
import numpy as np
from controller import Robot, Emitter
import robot_control, utils
import numpy as np
import matplotlib.pyplot as plt


INF = float("inf")


def encode_base64(input_string):
    return base64.b64encode(input_string.encode("utf-8")).decode("utf-8")


def decode_base64(base64_string):
    return base64.b64decode(base64_string.encode("utf-8")).decode("utf-8")


def show_grid(
    ax,
    density,
    orientation,
    current_grid_pos,
    goal,
    path,
    trajectory,
    grid,
):

    ax.clear()
    ax.plot(
        np.zeros((density, density)),
        "wo",
        markersize=0,
    )

    p = np.array(path)
    if len(p) > 0:
        ax.plot(
            p[:, 0],
            p[:, 1],
            "go",
            label="Waypoints",
            markersize=5,
        )

    obstacles = np.array(np.where(grid == 1)).transpose()
    ax.plot(
        obstacles[:, 0],
        obstacles[:, 1],
        "rx",
        label="Obstacles",
        markersize=5,
    )

    ax.plot(
        goal[0],
        goal[1],
        "gx",
        label="Target",
        markersize=10,
    )

    prev_step = None
    for step in trajectory[::-1]:
        if prev_step is None:
            prev_step = step
        ax.plot(
            (step[0], prev_step[0]),  # X values
            (step[1], prev_step[1]),  # Y values
            linestyle="-",  # Solid line connecting points
            color="b",  # Optional: line color, e.g., blue
        )
        prev_step = step

    ax.plot(
        current_grid_pos[0],
        current_grid_pos[1],
        "bo",
        label="Robot",
        markersize=7,
    )
    length = 1
    O_radians = np.deg2rad(orientation)
    x, y = current_grid_pos[0], current_grid_pos[1]
    x_end = x + length * np.cos(O_radians)
    y_end = y + length * np.sin(O_radians)

    # Plot the arrow indicating the direction
    ax.quiver(
        x,
        y,
        x_end - x,
        y_end - y,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="r",
    )

    ax.set_aspect("equal")
    # ax.axis('scaled')
    ax.legend()
    plt.xticks(
        np.arange(-0.5, density, 1),
        labels=[],
    )
    plt.yticks(
        np.arange(-0.5, density, 1),
        fontsize=8,
    )
    ax.set_xlim(-0.5, density)
    ax.set_ylim(-0.5, density)
    ax.grid(True)

    ax.figure.savefig(
        f"renders.png",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )


class AgentContoller:

    robot: Robot
    completed: bool = False
    initial_position = [0, 0]
    customData: dict = {}

    def __init__(
        self,
        robot: Robot,
    ):

        self.robot = robot

        self.control = robot_control.RobotControl(
            robot,
            speed=6,
        )

        self.name = robot.getName()
        targetStr = self.robot.getCustomData()
        self.customData = json.loads(decode_base64(targetStr))
        self.target = self.customData["target"]

    def get_alignment(self, target_pos):
        v = utils.get_alignment_to_target(
            self.control.orientation,
            [
                self.control.x,
                self.control.y,
            ],
            target_pos,
        )

        v = v % 360
        if v > 180:
            v = -(180 - (v - 180))

        return v / 180

    def get_distance_to_target(self, target_pos):

        travel_distance = utils.calculate_distance(self.initial_position, target_pos)
        current_distance = utils.calculate_distance(
            (self.control.x, self.control.y), target_pos
        )

        alignment = self.get_alignment(target_pos)
        factor = -1 if np.abs(alignment) < 0.5 else 1

        if travel_distance != 0:
            d = (current_distance) * (0.5 / travel_distance)
        else:
            d = 0
        if d < -1:
            d = -1
        if d > 1:
            d = 1

        return d * factor

    def move_to_target(self):

        self.control.sync()
        x, y = self.control.x, self.control.y
        initial_position = [x, y]
        target = self.target

        kind = 4
        density = 30
        scale = 1 / density
        grid = np.zeros((density, density))
        trajectory = []

        offset = 0.5

        current_grid_pos, robot_rem = utils.apply_transformation(
            initial_position,
            offset=offset,
            scale=scale,
        )
        goal, target_rem = utils.apply_transformation(
            target,
            offset=offset,
            scale=scale,
        )

        
        path = utils.a_star(current_grid_pos, goal, grid, kind)

        if len(path) == 0:
            print("No path found")
            return

        subGoal = path.pop(0)
        self.control.speed = 1

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        achievedSubGoal = current_grid_pos

        render = lambda: show_grid(
            ax,
            density,
            self.control.orientation,
            current_grid_pos,
            goal,
            path,
            trajectory,
            grid,
        )

        while subGoal is not None:
            # render()
            subTarget = utils.revert_transformation(
                subGoal,
                offset,
                scale,
                remainders=target_rem,
            )

            obstacles = self.control.detect_obstacles()

            self.control.sync()
            alignment = self.get_alignment(subTarget)
            alin_thres = 5.0 / 180
            x, y = self.control.x, self.control.y

            current_grid_pos, robot_rem = utils.apply_transformation(
                [x, y],
                offset=offset,
                scale=scale,
            )

            posThres = 0.02
            _, posError = utils.is_coordinate_equal(
                subTarget,
                [x, y],
            )

            hasReachedSubGoal = (
                posError < posThres or subGoal == current_grid_pos
            ) and grid[subGoal] != 1

            if np.abs(alignment) >= 170 / 180:
                print("reverse")
                self.control.move(-1, -1)
                continue

            if np.abs(alignment) >= alin_thres and not hasReachedSubGoal:
                factor = alignment / np.abs(alignment)
                self.control.move(-1 * factor, 1 * factor)
                continue

            if hasReachedSubGoal:
                if len(path) == 0:
                    subGoal = None
                    continue
                grid[subGoal] = -1
                trajectory.append(subGoal)
                achievedSubGoal = subGoal
                subGoal = path.pop(0)
                continue

            if np.sum(obstacles) > 0:

                self.control.motor_stop()

                if subGoal != goal and grid[subGoal] != 1:
                    if grid[subGoal] == -1:
                        print("will override past position", subGoal)
                    grid[subGoal] = obstacles[0]

                # sub_moves = utils.generate_sub_moves(current_grid_pos, subGoal)

                # if len(sub_moves) > 1:
                #     sub_moves = np.array(sub_moves)

                #     sub_moves_grid = grid[sub_moves[:, 0], sub_moves[:, 1]]

                #     if sub_moves_grid.sum() > 0:
                #         grid[sub_moves[:, 0], sub_moves[:, 1]] = 1
                #         print("Added sub moves ", sub_moves)

                path = utils.a_star(achievedSubGoal, goal, grid, kind)

                if len(path) == 0:
                    print("No path found")
                    break

                subGoal = path.pop(0)
                while subGoal == achievedSubGoal and len(path) > 0:
                    subGoal = path.pop(0)

                print("avoid")
                continue

            self.control.move(1, 1)

        render()
        self.control.motor_stop()
        e: Emitter = self.robot.getDevice("emitter")
        e.send(self.name)


print("Running Agent")
robot = Robot()
m = AgentContoller(robot)
m.move_to_target()
