import json
import numpy as np
import pandas as pd
from controller import Supervisor
import numpy as np
from scipy.optimize import linear_sum_assignment
import json
from controller import Supervisor, Emitter, Receiver
import threading


robots_pos_override = [
    [0, 0],
    # [0.2, 0],
]

targets_pos_override = [
    [-0.05, 0],
    # [0.4, -0.4],
]

obstacle_pos_override = [
    [0.2, 0],
    # [0.4, -0.4],
]

TIMEOUT = 60 * 5

df = pd.DataFrame(
    columns=[
        "n_robots",
        "approach",
        "mean_pos_error",
        "max_time",
        "total_energy",
        "timed_out",
    ]
)


def generate_random_coordinates(x_min, x_max, y_min, y_max, n=1, threshold=0.08):
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
        x_new = np.random.uniform(x_min, x_max)
        y_new = np.random.uniform(y_min, y_max)
        new_point = np.array([x_new, y_new])

        # Check distance to all existing points
        if all(
            np.linalg.norm(new_point - existing_point) > threshold
            for existing_point in coordinates
        ):
            coordinates.append(new_point)

    return np.array(coordinates)


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

    def __init__(
        self,
        supervisor: Supervisor,
        n_robots: int,
        arena_size: np.ndarray,
        n_obstacles: int = 0,
        **kwargs,
    ):
        self.supervisor = supervisor
        self.n_robots = n_robots
        self.arena_size = arena_size

        self.timestep = int(self.supervisor.getBasicTimeStep())

        x_min, x_max = np.asarray((-1, 1)) * (arena_size[0] / 2)
        y_min, y_max = np.asarray((-1, 1)) * (arena_size[1] / 2)

        cords = generate_random_coordinates(
            x_min, x_max, y_min, y_max, 2 * n_robots + n_obstacles
        )

        self.robot_positions = cords[:n_robots, :]

        self.target_positions = cords[n_robots : 2 * n_robots, :]

        if n_obstacles > 0:
            self.obstacle_positions = cords[2 * n_robots :, :]

        if kwargs.get("override_positions") == True:
            self.robot_positions = np.array(robots_pos_override)
            self.target_positions = np.array(targets_pos_override)
            self.obstacle_positions = np.array(obstacle_pos_override)

        if kwargs.get("cache_positions") == True:
            with open("positions_cache.py", "a") as fd:
                fd.writelines(
                    [
                        "robots = " + json.dumps(self.robot_positions.tolist()) + "\n",
                        "targets = "
                        + json.dumps(self.target_positions.tolist())
                        + "\n\n",
                    ]
                )

        for index, target in enumerate(self.target_positions, 0):
            self.unsatified_targets[index] = target.tolist()

        self.n_robots = len(self.robot_positions)

        floor = self.supervisor.getFromDef("FLOOR")

        floor.getField("floorSize").setSFVec2f(list(arena_size))

        self.step()

        self.started_at = self.supervisor.getTime()

    def step(self):
        return self.supervisor.step(self.timestep)

    def reset(self):
        for rowCount in range(self.n_robots):
            node = self.supervisor.getFromDef(f"robot{rowCount}")
            if node is not None:
                node.remove()
            node = self.supervisor.getFromDef(f"target{rowCount}")
            if node is not None:
                node.remove()
            
        for rowCount in range(len(self.obstacle_positions)):
            node = self.supervisor.getFromDef(f"obstacle{rowCount}")
            if node is not None:
                node.remove()
        
        self.step()
        self.started_at = self.supervisor.getTime()

    def insert_robots_and_targets(self):

        root_node = self.supervisor.getRoot()
        children_field = root_node.getField("children")
        controllerFile = "agent_controller"

        rowCount = 0
        for (x, y), (tx, ty) in zip(
            self.robot_positions, self.target_positions
        ):

            children_field.importMFNodeFromString(
                -1,
                f"""
                DEF robot{rowCount}  AgentProto {{
                name "robot{rowCount}"
                translation {x} {y} 0.01
                rotation 0 0 1 {np.deg2rad(0)}
                controller "{controllerFile}"
                }}
                """,
            )
            
            children_field.importMFNodeFromString(
                -1,
                f"""
                    DEF target{rowCount} TargetProto {{
                        name "target{rowCount}"
                        translation {tx} {ty} 0.1
                    }}
                """,
            )
            rowCount += 1

        rowCount = 0
        for obx, oby in self.obstacle_positions:
            children_field.importMFNodeFromString(
                -1,
                f"""
                    DEF obstacle{rowCount} ObstacleProto {{
                        name "obstacle{rowCount}"
                        translation {obx} {oby} 0.1
                    }}
                """,
            )
            
            rowCount += 1
        self.step()

    def send_task_to_robot(
        self,
        index: int,
        tag: str,
        data,
    ):
        emitter: Emitter = self.supervisor.getDevice("emitter")

        emitter.send(
            json.dumps({"name": f"robot{index}", "tag": tag, "data": data}),
        )

    def receive_data(self) -> str:
        receiver: Receiver = self.supervisor.getDevice("receiver")
        receiver.enable(1)
        if receiver.getQueueLength() > 0:
            data = receiver.getString()
            receiver.nextPacket()
            return data

        return None

    def await_completion(self):
        global df, TIMEOUT

        energies = []
        durations = []
        posErrors = []
        while True:

            currentTime = self.supervisor.getTime() - self.started_at

            if currentTime > TIMEOUT:
                total_energy = np.sum([0, *energies])
                max_time = np.max([0, *durations])
                posError = np.mean([0] if len(posErrors) == 0 else posErrors)
                new_row = {
                    "n_robots": [self.n_robots],
                    "approach": [self.approach],
                    "mean_pos_error": [round(posError, 3)],
                    "max_time": [round(currentTime, 3)],
                    "total_energy": [round(total_energy, 3)],
                    "timed_out": [True],
                }
                df2 = pd.DataFrame(new_row)
                df = df2 if df.empty else pd.concat([df, df2], ignore_index=True)
                break

            self.step()
            data = self.receive_data()

            if data == None:
                continue

            dataJson: dict = json.loads(data)

            if type(dataJson) != dict:
                raise Exception("data json not of type dict")

            action = dataJson.get("action")
            body: dict = dataJson.get("body")

            if action == "on_complete" and type(body) == dict:
                positionErrors = body["position_errors"]
                duration = body["duration"]
                energy = body["energy"]

                posErrors.extend(positionErrors)
                durations.append(duration)
                energies.append(energy)

                if len(posErrors) == self.n_robots:
                    total_energy = np.sum(energies)
                    max_time = np.max(durations)
                    posError = np.mean(posErrors)

                    new_row = {
                        "n_robots": [self.n_robots],
                        "approach": [self.approach],
                        "mean_pos_error": [round(posError, 3)],
                        "max_time": [round(max_time, 3)],
                        "total_energy": [round(total_energy, 3)],
                        "timed_out": [False],
                    }
                    df2 = pd.DataFrame(new_row)
                    df = df2 if df.empty else pd.concat([df, df2], ignore_index=True)
                    break

    def run_hungarian(self) -> np.ndarray:
        self.approach = "hungarian"

        self.insert_robots_and_targets()
        # Calculate the cost matrix (Euclidean distances between robots and targets)
        cost_matrix = np.linalg.norm(
            self.robot_positions[:, None, :] - self.target_positions[None, :, :], axis=2
        )

        # Apply the Hungarian algorithm to find the optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Reorder target positions based on the assignment for visualization
        assigned_target_positions = self.target_positions[col_indices]
        assignments = assigned_target_positions

        for index, task in enumerate(assignments, 0):
            self.send_task_to_robot(index, "hungarian", list(task))

        self.await_completion()

    def run_mean_field(self) -> np.ndarray:
        self.approach = "mean_field"

        self.insert_robots_and_targets()

        self.supervisor.setCustomData(json.dumps(self.unsatified_targets))

        for index, _ in self.unsatified_targets.items():
            self.send_task_to_robot(index, "mean_field", self.unsatified_targets)

        self.await_completion()

    def run_market_based(self) -> np.ndarray:
        self.approach = "market"

        self.insert_robots_and_targets()

        n_tasks = self.n_robots

        bids = np.linalg.norm(
            self.robot_positions[:, None, :] - self.target_positions[None, :, :], axis=2
        )

        # Assign tasks to robots
        task_assignments = {}
        for task_index in range(n_tasks):
            # Find robot with the lowest bid for each task
            robot_index = np.argmin(bids[:, task_index])
            task_assignments[task_index] = robot_index
            # Increase the bid to a very high value to simulate task assignment (task cannot be assigned again)
            bids[robot_index, :] = np.inf

        for task, robot in task_assignments.items():
            self.send_task_to_robot(
                robot, "market_based", self.target_positions[task].tolist()
            )

        self.await_completion()

    def run_threshold(self) -> np.ndarray:
        self.approach = "threshold"

        self.insert_robots_and_targets()

        n_tasks = self.n_robots

        distance_threshold = (
            3.0  # Maximum distance for a robot to be eligible for a task
        )

        task_thresholds = np.full(
            n_tasks, distance_threshold
        )  # Threshold for each task

        # Calculate distances from each robot to each task
        distances = np.linalg.norm(
            self.robot_positions[:, None, :] - self.target_positions[None, :, :], axis=2
        )

        eligible = distances < task_thresholds

        task_assignments = {}

        for task_index in range(n_tasks):
            eligible_robots = np.where(eligible[:, task_index])[0]
            if eligible_robots.size > 0:
                minIndex = np.argmin(distances[eligible_robots, task_index])

                closest_robot = eligible_robots[minIndex]

                assignments = task_assignments.get(closest_robot, [])

                assignments.append(task_index)

                task_assignments[closest_robot] = assignments

        for robot, task in task_assignments.items():

            taskList: list = self.target_positions[task].tolist()

            self.send_task_to_robot(robot, "threshold_based", taskList)

        self.await_completion()


def simulate_task_assignments(
    sup: Agents_Supervisor,
    csv_file="eval.csv",
    save_csv=False,
):

    sup.run_hungarian()
    sup.reset()
    sup.run_mean_field()
    sup.reset()
    sup.run_market_based()
    sup.reset()
    sup.run_threshold()
    sup.reset()

    if save_csv:
        df.to_csv(csv_file)
    supervisor.step()


robot = Supervisor()

supervisor = Agents_Supervisor(
    robot,
    10,
    np.asarray([.7, .5]),
    n_obstacles=0,
    cache_positions=not True,
    override_positions= not True,
)
simulate_task_assignments(
    supervisor,
    csv_file="eval.csv",
    save_csv=True,
)
