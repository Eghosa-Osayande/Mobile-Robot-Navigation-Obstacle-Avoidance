import base64
import json
import numpy as np
import pandas as pd
from controller import Supervisor, Receiver
import numpy as np
import json
from PIL import Image, ImageFont, ImageDraw


def encode_base64(input_string):
    return base64.b64encode(input_string.encode("utf-8")).decode("utf-8")


def decode_base64(base64_string):
    return base64.b64decode(base64_string.encode("utf-8")).decode("utf-8")


def generate_random_coordinates(
    x_min, x_max, y_min, y_max, n=1, threshold=0.08, padding=0
):
    """
    Generate random (x, y) coordinates within specified bounds, ensuring no two points are closer than the threshold.

    Parameters:
    - n: int, number of coordinates to generate
    - x_min: float, minimum value for x coordinates
    - x_max: float, maximum value for x coordinates
    - y_min: float, minimum value for y coordinates
    - y_max: float, maximum value for y coordinates
    - threshold: float, minimum Euclidean distance between any two points

    Returns:
    - coordinates: array of shape (n, 2) containing generated (x, y) coordinates
    """
    coordinates = []
    while len(coordinates) < n:
        # Generate a new candidate point
        x_new = np.random.uniform(x_min + padding, x_max - padding)
        y_new = np.random.uniform(y_min + padding, y_max - padding)
        new_point = np.array([x_new, y_new])

        # Check distance to all existing points
        if all(
            np.linalg.norm(new_point - existing_point) > threshold
            for existing_point in coordinates
        ):
            coordinates.append(new_point)
    r = np.array(coordinates)
    # np.random.shuffle(r)
    return r


def sentence_to_array(sentence, size):
    font = ImageFont.load_default()
    image = Image.new("1", size, 1)  # '1' for 1-bit pixels, black and white
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), sentence, font=font)
    led_array = np.array(image)
    led_array = np.where(led_array == 1, 0, 1)
    return led_array[::-1, :]


def sentence_to_targets(
    sentence,
    target_size=[1, 1],
    origin_offest=[0, 0],
):
    size = [50, 12]
    unit_map = sentence_to_array(sentence, size)
    targets = []

    for index, row in enumerate(unit_map):
        height_offset = index * target_size[1]
        height = height_offset + (target_size[1] / 2) + origin_offest[1]
        for index, column in enumerate(row):
            width_offset = index * target_size[0]
            width = width_offset + (target_size[0] / 2) + origin_offest[0]
            if column:
                targets.append([width, height])

    return np.array(targets)


class Agents_Supervisor:

    supervisor: Supervisor
    timestep: int

    n_robots: int
    arena_size: np.ndarray

    robot_positions: np.ndarray

    target_positions: np.ndarray

    obstacle_positions: np.ndarray = np.array([])

    unsatified_targets: dict = {}

    approach: str = "None"
    started_at: float = 0
    show_targets = True

    def __init__(
        self,
        supervisor: Supervisor,
        n_robots: int,
        arena_size: np.ndarray,
        n_obstacles: int = 0,
        show_targets=True,
        show_obstacles=True,
        **kwargs,
    ):
        self.supervisor = supervisor
        self.n_robots = n_robots
        self.arena_size = arena_size
        self.show_targets = show_targets
        self.show_obstacles = show_obstacles

        self.timestep = int(self.supervisor.getBasicTimeStep())

        x_min, x_max = np.asarray((-1, 1)) * (arena_size[0] / 2)
        y_min, y_max = np.asarray((-1, 1)) * (arena_size[1] / 2)

        cords = generate_random_coordinates(
            x_min,
            x_max,
            y_min,
            y_max,
            2 * n_robots + n_obstacles,
            padding=0.05,
        )

        self.robot_positions = cords[:n_robots, :]

        self.target_positions = cords[n_robots : 2 * n_robots, :]

        if n_obstacles > 0:
            self.obstacle_positions = cords[2 * n_robots :, :]

        if (robot_positions := kwargs.get("robot_positions")) is not None:
            print(robot_positions)
            self.robot_positions = np.array(robot_positions)

        if (target_positions := kwargs.get("target_positions")) is not None:
            self.target_positions = np.array(target_positions)

        if (obstacle_positions := kwargs.get("obstacle_positions")) is not None:
            self.obstacle_positions = np.array(obstacle_positions)

        if kwargs.get("cache_positions") == True:
            with open("positions_cache.py", "a") as fd:
                fd.writelines(
                    [
                        "robot_positions = "
                        + json.dumps(self.robot_positions.tolist())
                        + "\n",
                        "target_positions = "
                        + json.dumps(self.target_positions.tolist())
                        + "\n\n",
                    ]
                )

        self.n_robots = len(self.robot_positions)

        floor = self.supervisor.getFromDef("FLOOR")

        floor.getField("floorSize").setSFVec2f(list(arena_size))

        self.step()

        self.started_at = self.supervisor.getTime()

    def step(self):
        return self.supervisor.step(self.timestep)

    def reset(self):
        for rowCount in range(len(self.robot_positions)):
            node = self.supervisor.getFromDef(f"robot{rowCount}")
            if node is not None:
                node.remove()
                self.step()
        if self.show_targets:
            for rowCount in range(len(self.target_positions)):
                node = self.supervisor.getFromDef(f"target{rowCount}")
                if node is not None:
                    node.remove()
                    self.step()

        for rowCount in range(len(self.obstacle_positions)):
            node = self.supervisor.getFromDef(f"obstacle{rowCount}")
            if node is not None:
                node.remove()
                self.step()

        self.step()
        self.started_at = self.supervisor.getTime()

    def insert_robots_and_targets(self, **kwargs):

        root_node = self.supervisor.getRoot()
        children_field = root_node.getField("children")
        controllerFile = "agent_controller"

        for rowCount, ((x, y), (tx, ty)) in enumerate(
            zip(self.robot_positions, self.target_positions)
        ):

            children_field.importMFNodeFromString(
                -1,
                f"""
                DEF robot{rowCount}  AgentProto {{
                name "robot{rowCount}"
                translation {x} {y} 0.01
                rotation 0 0 1 {np.deg2rad(0)}
                controller "{controllerFile}"
                customData "{encode_base64(json.dumps({'target':[tx,ty],**kwargs}))}"
                }}
                """,
            )

            if self.show_targets:

                for rowCount, (tx, ty) in enumerate(self.target_positions):

                    children_field.importMFNodeFromString(
                        -1,
                        f"""
                            DEF target{rowCount} TargetProto {{
                                name "target{rowCount}"
                                translation {tx} {ty} 0.1
                            }}
                        """,
                    )
        if self.show_obstacles == True:
            for rowCount, (obx, oby) in enumerate(self.obstacle_positions):
                children_field.importMFNodeFromString(
                    -1,
                    f"""
                        DEF obstacle{rowCount} ObstacleProto {{
                            name "obstacle{rowCount}"
                            translation {obx} {oby} 0.1
                        }}
                    """,
                )

        self.step()

    def await_completion(self):
        r: Receiver = self.supervisor.getDevice("receiver")
        r.enable(1000)
        packets = []
        while len(packets) < len(self.robot_positions):
            self.supervisor.step()
            if r.getQueueLength() > 0:
                data = r.getString()
                packets.append(data)
                r.nextPacket()
        print("Done")


# targets = sentence_to_targets("O", origin_offest=(-0.5, -0.5), target_size=(0.1, 0.1))

arena_size = [1, 1]
robot = Supervisor()

seeds = [45, 46, 48, 49]
scales = [30]

for seed in seeds:
    # seed=48
    kind = 8 # 4 or 8 way movement
    for scale in scales:
        np.random.seed(seed)
        supervisor = Agents_Supervisor(
            robot,
            n_robots=1,
            arena_size=np.asarray(arena_size),
            n_obstacles=20,
            cache_positions=False,
            # target_positions=[(0.4,0)],
            # robot_positions=[(-0.4,0)],
            # obstacle_positions=[(0,0)]
        )

        supervisor.insert_robots_and_targets(
            seed=seed,
            kind=kind,
            scale=scale,
        )
        supervisor.await_completion()
        robot.exportImage(f"final-{seed}-k{kind}-s{scale}.png", 100)
        supervisor.reset()
        print("seed ", seed)
        # break
