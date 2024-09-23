import json
import numpy as np
from controller import Robot, Emitter
import robot_control, utils


INF = float("inf")


class AgentContoller:

    robot: Robot
    completed: bool = False
    initial_position = [0, 0]

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
        self.target = json.loads(targetStr)

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

        density = 16
        grid = np.zeros((density, density))
        scale = 1 / density
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

        path = utils.a_star(current_grid_pos, goal, grid)

        if len(path) == 0:
            print("No path found")
            return

        subGoal = path.pop(0)
        self.control.speed = 1

        def show_grid():
            g = grid.astype(str)

            g = np.where(g == "0.0", "-", g)
            g = np.where(g == "1.0", "X", g)
            for cc, p in enumerate(
                path,
            ):
                if g[p] == "-":
                    g[p] = f"{cc+2}"
                    g[p] = f"*"
                    pass
            g[current_grid_pos] = (
                "O" if g[current_grid_pos] == "-" else g[current_grid_pos]
            )

            g[goal] = "+"

            g = np.rot90(g, k=1)
            print(g)

        while subGoal is not None:
            # show_grid()
            subTarget = utils.revert_transformation(
                subGoal,
                offset,
                scale,
                remainders=target_rem,
            )
            obstacles = self.control.detect_obstacles()
            self.control.sync()
            alignment = self.get_alignment(subTarget)
            alin_thres = 10 / 180

            if np.abs(alignment) > alin_thres:
                factor = alignment / np.abs(alignment)
                self.control.move(-1 * factor, 1 * factor)
                continue

            x, y = self.control.x, self.control.y

            posThres = 0.02
            _, posError = utils.is_coordinate_equal(
                subTarget,
                [x, y],
            )
            if posError < posThres or subGoal == current_grid_pos:

                if len(path) == 0:
                    subGoal = None
                    continue
                subGoal = path.pop(0)
                continue

            # print(
            #     "go to ",
            #     subGoal,
            #     " I am at ",
            #     current_grid_pos,
            #     obstacles,
            # )

            if np.sum(obstacles) > 0:
                # self.control.motor_stop()
                self.control.sync()
                x, y = self.control.x, self.control.y
                current_grid_pos, robot_rem = utils.apply_transformation(
                    [x, y],
                    offset=offset,
                    scale=scale,
                )
                edges = [
                    (0, -1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                    (0, 1),
                    (-1, 1),
                    (-1, 0),
                    (-1, -1),
                ]

                orientation = self.control.orientation
                orientation += 22.5
                start_index = int((orientation // 45))

                for obs_index, edge_index in enumerate(
                    range(start_index, start_index + 5), 0
                ):
                    edge_index = edge_index % len(edges)
                    edge = edges[edge_index]
                    xx, yy = (
                        current_grid_pos[0] + edge[0],
                        current_grid_pos[1] + edge[1],
                    )
                    if (
                        0 <= xx < grid.shape[0]
                        and 0 <= yy < grid.shape[1]
                        and (xx, yy) != goal
                        and grid[xx, yy] == 0
                    ):
                        grid[xx, yy] = obstacles[obs_index]

                sub_moves = utils.generate_sub_moves(current_grid_pos, subGoal)
                if len(sub_moves) > 1:
                    sub_moves = np.array(sub_moves)

                    sub_moves_grid = grid[sub_moves[:, 0], sub_moves[:, 1]]

                    if sub_moves_grid.sum() > 0:
                        grid[sub_moves[:, 0], sub_moves[:, 1]] = 1
                        print("altered")

                path_array = np.array([subGoal, *path])
                path_grid = grid[path_array[:, 0], path_array[:, 1]]

                # show_grid()

                if path_grid.sum() > 0:
                    print("Avoid")

                    path = utils.a_star(current_grid_pos, goal, grid)

                    if len(path) == 0:
                        print("No path found")
                        break

                    subGoal = path.pop(0)
                    # show_grid()
                    # return
                    continue

            self.control.move(1, 1)

        self.control.motor_stop()
        e: Emitter = self.robot.getDevice("emitter")
        e.send(self.name)


print("Running RL Agent")
robot = Robot()
m = AgentContoller(robot)
m.move_to_target()
