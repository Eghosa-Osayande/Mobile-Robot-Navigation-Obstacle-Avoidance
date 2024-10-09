import base64
import json

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
    orientation,
    current_grid_pos,
    goal,
    path,
    grid,
    fileName="render.png",
):  

    ax.clear()

    p = np.array(path)
    if len(p) > 0:
        ax.plot(
            p[:, 0],
            p[:, 1],
            "go",
            label="Waypoints",
            markersize=5,
        )
    density = 0

    labels = {
        1: "Obstacles",
        -1: "Trajectory",
    }
    markers = {1: "rx", -1: "b."}
    for point, value in grid.items():

        ax.plot(
            point[0],
            point[1],
            markers.get(value),
            label=labels.get(value),
            markersize=5,
        )
        labels.pop(value, None)
        density = np.max(np.abs([density, *point, *goal, *current_grid_pos]))

    ax.plot(
        goal[0],
        goal[1],
        "gx",
        label="Target",
        markersize=10,
    )

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
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    start, end = -(density + 0.5), (density + 0.5)

    plt.xticks(
        np.arange(start, end, 1),
        labels=[],
    )
    plt.yticks(
        np.arange(start, end, 1),
        fontsize=8,
    )
    # plt.tight_layout()

    ax.set_xlim(
        start,
        end,
    )
    ax.set_ylim(
        start,
        end,
    )

    ax.grid(True)

    

    ax.figure.savefig(
        fileName,
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
            speed=0.1,
        )

        self.name = robot.getName()
        targetStr = self.robot.getCustomData()
        self.customData = json.loads(decode_base64(targetStr))
        self.target = self.customData["target"]
        self.seed = str(self.customData.get("seed"))
        self.kind = int(self.customData.get("kind"))
        self.scale = float(self.customData.get("scale"))

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
        return v

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
        # x,y <- get robot current position
        x, y = self.control.x, self.control.y
        initial_position = [x, y]
        target = self.target

        kind = self.kind

        # scaale <- set the scale factor for mapping robot position to grid index
        scale = 1 / self.scale

        # gridMap <- initialise map to hold robot experience at explored grid index
        gridMap = {}
        limits = [(-15, 15), (-15, 15)]

        offset = 0.0

        # current_grid_pos <- convert the robot starting position to equivalent grid index
        current_grid_pos, robot_rem = utils.point_to_index(
            initial_position,
            offset=offset,
            scale=scale,
        )

        # goal <- convert the target position to equivalent grid index
        goal, target_rem = utils.point_to_index(
            target,
            offset=offset,
            scale=scale,
        )

        # path <- find initial path using a star from start index to target index
        path = utils.a_star(
            current_grid_pos,
            goal,
            gridMap,
            kind,
            # limits=limits,
        )

        # if len of path is zero END

        if len(path) == 0:
            print("No path found")
            return

        # set the robots current grid index to -1
        lastAchievedSubGoal = path.pop(0)

        # subGoal <- get the first sub goal from the a star path
        subGoal = path.pop(0)
        # subTarget <- convert subGoal to in cartesion cordinte
        subTarget = utils.index_to_point(
            subGoal,
            offset,
            scale,
            remainders=target_rem,
        )

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        render = lambda fileName="": show_grid(
            ax,
            self.control.orientation,
            current_grid_pos,
            goal,
            path,
            gridMap,
        )

        # while sub_goal has a value
        fileName=f"final-{self.seed}-k{kind}-s{self.scale}"

        render(fileName=fileName)
        timestep=0
        while subGoal is not None:
            timestep+=1
            if timestep%10 ==0:
                render(fileName=fileName)
            self.control.sync()
            # x,y <- get robot current position
            x, y = self.control.x, self.control.y

            # current_grid_pos <- convert x,y to grid index
            current_grid_pos, robot_rem = utils.point_to_index(
                [x, y],
                offset=offset,
                scale=scale,
            )

            posThres = 0.02
            _, posError = utils.is_coordinate_equal(
                subTarget,
                [x, y],
            )

            # check if robot has reached subGoal
            hasReachedSubGoal = (
                posError < posThres or subGoal == current_grid_pos
            ) and gridMap.get(subGoal) != 1

            # if yes

            if hasReachedSubGoal:
                # if len of path is zero
                # assign subGoal to no value
                if len(path) == 0:
                    subGoal = None
                    continue

                # set the grid index to -1
                if True:  # and gridMap.get(subGoal) is None:
                    gridMap[subGoal] = -1

                # lastAchievedSubGoal <- subgoal
                lastAchievedSubGoal = subGoal

                # subGoal <- pop the next subgoal from the path list
                subGoal = path.pop(0)

                # subTarget <- convert subGoal to in cartesion cordinte
                subTarget = utils.index_to_point(
                    subGoal,
                    offset,
                    scale,
                    remainders=target_rem,
                )

                continue

            # alignment <- get robot alignment to target
            alignment = self.get_alignment(subTarget)

            # if alignment < 0 turn right
            # if alignment > 0 turn left
            # continue
            # print(current_grid_pos,subGoal,alignment)

                

            if alignment > 10:
                if alignment >= 170 and alignment <=  190:
                    print("reverse")
                    self.control.move(-1, -1)
                    continue
                factor = 1
                if alignment > 180:
                    factor = -1
                self.control.speed = 1
                self.control.move(-1 * factor, 1 * factor)
                continue

            # obstacles <- detect obstacles
            obstacles = self.control.detect_obstacles()

            # if obstacle in forward path
            if (
                self.control.isReversing == True
                and obstacles[1] == 1
                or self.control.isReversing == False
                and obstacles[0] == 1
            ):
                # stop robot
                self.control.motor_stop()

                # subGoal <- set grid cell with index == subGoal to 1
                if subGoal != goal:  # and gridMap.get(subGoal) is None:
                    # if gridMap.get(subGoal) == -1:
                    #     print("will override past position", subGoal)

                    gridMap[subGoal] = 1

                # path <- recompute astar path without grid cells with value == 1
                path = utils.a_star(
                    lastAchievedSubGoal,
                    goal,
                    gridMap,
                    kind,
                    # limits=limits,
                )

                # if len of path == 0 , break, no path found

                if len(path) == 0:
                    print("No path found")
                    break

                # subGoal <- pop the next subgoal from the path list
                subGoal = path.pop(0)

                # continue

                while subGoal == lastAchievedSubGoal and len(path) > 0:
                    subGoal = path.pop(0)

                # subTarget <- convert subGoal to in cartesion cordinte
                subTarget = utils.index_to_point(
                    subGoal,
                    offset,
                    scale,
                    remainders=target_rem,
                )
                print(current_grid_pos, subGoal, gridMap.get(subGoal))
                render(fileName=fileName)
                print("avoid")
                continue

            # move forward
            if not self.control.isReversing:
                self.control.speed = 5
                self.control.move(1, 1)
        # stop robot
        render(fileName=fileName)
        self.control.motor_stop()
        e: Emitter = self.robot.getDevice("emitter")
        e.send(self.name)


print("Running Agent")
robot = Robot()
m = AgentContoller(robot)
m.move_to_target()
