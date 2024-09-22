import json
import numpy as np
from stable_baselines3 import A2C
from controller import (
    DistanceSensor,
    Supervisor,
    Field,
    Motor,
    Receiver,
    GPS,
    Compass,
    Emitter,
    Supervisor,
)
import robot_control, utils


COORDINATE_MATCHING_ACCURACY = 0.01


INF = float("inf")


class BotClient:
    def __init__(self, field: Field):
        self.field = field

    def delete_position(self, position_id):
        pos = json.loads(self.field.getSFString())
        _ = pos.pop(str(position_id))
        self.field.setSFString(json.dumps(pos))

    def get_positions(self):
        pos = json.loads(self.field.getSFString())
        return pos


class AgentContoller:

    timestep: float
    robot: Supervisor

    completed: bool = False
    client: BotClient
    started_at: float = 0
    position_errors = []
    receiver: Receiver
    emitter: Emitter
    initial_position = [0, 0]

    def __init__(
        self,
        robot: Supervisor,
    ):

        self.robot = robot
        timestep = int(robot.getBasicTimeStep())

        receiver: Receiver = robot.getDevice("receiver1")
        receiver.enable(1)
        self.receiver = receiver

        emitter: Emitter = robot.getDevice("emitter1")
        self.emitter = emitter

        sensor_tags = [
            "ps5",
            "ps6",
            "ps7",
            "ps0",
            "ps1",
            "ps2",
        ][::-1]

        self.distance_sensors = []
        for sensor in sensor_tags:
            ps: DistanceSensor = robot.getDevice(sensor)
            ps.enable(1)
            self.distance_sensors.append(ps)

        self.control = robot_control.RobotControl(
            robot,
            speed=6,
        )
        self.started_at = self.robot.getTime()
        self.name = robot.getName()

        botCustomData = self.robot.getFromDef("BOT").getField("customData")

        self.client = BotClient(botCustomData)
        density = 20
        self.grid = np.zeros((density, density))
        self.scale = 1 / density
        self.path = []
        self.model = A2C.load("a2c-v3-30000")
        self.move_model = A2C.load("a2c-move-to-target-v1-30000")
        self.turn_model = A2C.load("a2c-turn-to-target-cont-30000")
        self.free_spaces=set()

    def check_battery(self):
        pass

    def on_complete(self):
        self.control.motor_stop()

        energy_consumed = 1000

        duration = self.robot.getTime() - self.started_at

        self.send_data(
            {
                "action": "on_complete",
                "body": {
                    "name": self.name,
                    "duration": duration,
                    "energy": energy_consumed,
                    "position_errors": self.position_errors,
                },
            }
        )
        self.completed = True
        print(self.name, " stopped")

    def send_data(self, data):
        self.emitter.send(
            json.dumps(
                data,
            )
        )

    def recieve_data(self):
        if self.receiver.getQueueLength() > 0:
            raw = self.receiver.getString()
            self.receiver.nextPacket()
            data_json: dict = json.loads(raw)
            name = data_json["name"]
            if name == self.name:
                return data_json
        return None

    def process_data(self, data_json: str):
        name = data_json["name"]
        data = data_json["data"]
        tag = data_json["tag"]

        funcMap = {
            "hungarian": self.run_hungarian,
            "mean_field": self.run_mean_field,
            "market_based": self.run_market_based,
            "threshold_based": self.run_threshold_based,
        }

        func = funcMap.get(tag)

        if name == self.name and func != None:
            func(data)

    def avoid_obstacles(self) -> bool:
        weights = [-1, -0.6, 0.4, 0.9]
        weight = [0, 0, 0, 0]

        for i, sensor in enumerate(self.distance_sensors):

            value = sensor.getValue()

            if value > 100:
                weight[i] = 1

        if weight[1] == 0 and weight[2] == 0:
            return False
        if weight[0] == 1 or weight[3] == 1:
            return True
        if np.sum(weight) > 0:
            return True
        else:
            return False

    def detect_obstacles(self):
        weight = [0, 0, 0, 0, 0, 0]
        s = []

        for i, sensor in enumerate(self.distance_sensors):
            value = sensor.getValue()
            s.append(value)
            if value > 77:
                weight[i] = 1

        weight = tuple(weight)

        detected_obs = [0, 0, 0, 0, 0]

        front = weight[1:5]
        right = weight[0:2]
        left = weight[4:0]

        frontConditions = [
            (1, 1, 1, 1),
            (0, 1, 1, 0),
            (1, 0, 0, 1),
            (1, 1, 0, 0),
            (1, 1, 1, 0),
            (0, 1, 1, 1),
            (0, 0, 1, 1),
        ]

        rightConditions = [
            (1, 0),
            (0, 1),
            (1, 1),
        ]
        lefttConditions = [i[::-1] for i in rightConditions]

        if front in frontConditions or 1 in front:
            detected_obs[1] = 1
            detected_obs[2] = 1
            detected_obs[3] = 1

        if right in rightConditions:
            detected_obs[0] = 1
            detected_obs[1] = 1

        if left in lefttConditions:
            detected_obs[3] = 1
            detected_obs[4] = 1

        return detected_obs

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

    def predict_rl(
        self,
        target_pos,
    ):

        alignment = self.get_alignment(target_pos)
        d = self.get_distance_to_target(target_pos)
        alignment_obs = np.asarray([alignment])
        distance_obs = np.asarray([d])

        direction, _ = self.move_model.predict(distance_obs)
        direction = direction - 1

        alin_thres = 5 / 180

        # avoidedObstacle = self.avoid_obstacles()
        # if avoidedObstacle:
        #     self.control.move(0.5, -0.5)
        #     return

        tilt, _ = self.model.predict(alignment_obs)
        if alignment < -alin_thres:
            self.control.move(tilt[0], tilt[1])
        if alignment > alin_thres:
            self.control.move(tilt[0], tilt[1])

        if np.abs(alignment) > alin_thres:
            return

        if np.abs(alignment) > alin_thres:
            return

        self.control.move(direction, direction)

    def predict_manual(
        self,
        target_pos,
    ):

        alignment = self.get_alignment(target_pos)
        d = self.get_distance_to_target(target_pos)

        alin_thres = 10 / 180

        if alignment < -alin_thres:
            self.control.move(1, -1)
        if alignment > alin_thres:
            self.control.move(-1, 1)

        if np.abs(alignment) > alin_thres:
            return

        if d < 0:
            self.control.move(1, 1)
        if d > 0:
            self.control.move(-1, -1)

    def predict(
        self,
        target_pos,
    ):
        
        obstacles = self.detect_obstacles()
        offset = 0.5
        scale = self.scale
        grid = self.grid.copy()

        x, y = self.control.x, self.control.y

        pos = (int((x + offset) // scale), int((y + offset) // scale))
        
        remainders = ((x + offset) % scale, (y + offset) % scale)
        goal = goal1 = (
            int((target_pos[0] + offset) // scale),
            int((target_pos[1] + offset) // scale),
        )

        neighbours = [
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

        for obs_index, i in enumerate(range(start_index, start_index + 5), 0):
            i = i % len(neighbours)
            n = neighbours[i]
            xx, yy = pos[0] + n[0], pos[1] + n[1]
            if (
                0 <= xx < grid.shape[0]
                and 0 <= yy < grid.shape[1]
                and (xx, yy) != goal
                and (xx, yy) != pos
                and grid[xx, yy] == 0
                and (xx,yy) not in self.free_spaces
            ):
                grid[xx, yy] = obstacles[obs_index]
        
        if np.sum(obstacles)>0:
            self.control.motor_stop()
            
            self.path = utils.a_star(pos, goal, grid)
            if len(self.path) == 0:
                print("No path found")
                return

            g = grid.astype(str)
            g = np.where(g == "0.0", "_", g)
            g = np.where(g == "1.0", "x", g)
            for cc,p in enumerate(self.path):
                g[p] = f"{cc}"
            g[pos] = "o"
            g[goal1] = "+"
            g=np.rot90(g,k=1)

            # print(g)
          
            
            
            for i,goal in enumerate(self.path):
                xx = (goal[0] * scale) - offset + remainders[0]
                yy = (goal[1] * scale) - offset + remainders[1]
                self.path[i]=(xx,yy)
            self.path.pop(0)
            if len(self.path) > 0:
                target_pos = self.path.pop(0)
                print("Start Goto ",target_pos,)
                return target_pos

        self.predict_manual(target_pos)

    def run_hungarian(self, destinationCoordinates: list):

        if self.completed:
            return
        x, y = self.control.x, self.control.y

        self.initial_position = [x, y]
        dd=(destinationCoordinates[0],destinationCoordinates[1])
        destinationCoordinates=dd

        while True:

            x, y = self.control.x, self.control.y

            isEqual, positionError = utils.is_coordinate_equal(
                destinationCoordinates,
                [x, y],
                matching_accuracy=COORDINATE_MATCHING_ACCURACY,
            )
            if isEqual:
                if destinationCoordinates!=dd:
                    print(888)
                    destinationCoordinates=dd
                    if len(self.path)>0:
                        destinationCoordinates=self.path.pop(0)
                        print("Goto ",destinationCoordinates,)
                    
                    continue
                self.position_errors.append(positionError)
                break

            newT=self.predict(destinationCoordinates)
            if newT is not None:
                destinationCoordinates=newT

        self.on_complete()

    def run_mean_field(self, target_positionsMap):

        count = 1

        while True:
            if self.completed:
                return

            target_positionsMap = (
                target_positionsMap if count == 1 else self.client.get_positions()
            )
            target_positions = list(target_positionsMap.values())

            count += 1

            if len(target_positions) == 0:
                return

            x, y, _ = self.control.gps.getValues()
            current_position = np.array([x, y])

            target_positions = np.array(target_positions)

            distances = np.linalg.norm(target_positions - current_position, axis=1)

            nearest_target_index = np.argmin(distances)

            destinationCoordinates = target_positions[nearest_target_index]

            isEqual, positionError = utils.is_coordinate_equal(
                destinationCoordinates,
                [x, y],
                matching_accuracy=COORDINATE_MATCHING_ACCURACY,
            )

            if isEqual:
                self.position_errors.append(positionError)
                self.client.delete_position(
                    list(target_positionsMap.keys())[int(nearest_target_index)]
                )

                self.on_complete()
                break
            self.predict(destinationCoordinates)

    def run_market_based(self, destinationCoordinates: list):
        if self.completed:
            return

        while True:

            x, y, _ = self.control.gps.getValues()
            isEqual, positionError = utils.is_coordinate_equal(
                destinationCoordinates,
                [x, y],
                matching_accuracy=COORDINATE_MATCHING_ACCURACY,
            )
            if isEqual:
                self.position_errors.append(positionError)
                self.on_complete()
                break

            self.predict(destinationCoordinates)

    def run_threshold_based(self, destinationCoordinatesList: list):
        if self.completed:
            return

        for destinationCoordinates in destinationCoordinatesList:

            while True:
                x, y, _ = self.control.gps.getValues()

                isEqual, positionError = utils.is_coordinate_equal(
                    destinationCoordinates,
                    [x, y],
                    matching_accuracy=COORDINATE_MATCHING_ACCURACY,
                )

                if isEqual:
                    self.position_errors.append(positionError)
                    break

                self.predict(destinationCoordinates)

        self.on_complete()


print("Running RL Agent")
robot = Supervisor()

m = AgentContoller(robot)

while robot.step() != -1:
    data = m.recieve_data()
    if data != None:
        m.process_data(data)
