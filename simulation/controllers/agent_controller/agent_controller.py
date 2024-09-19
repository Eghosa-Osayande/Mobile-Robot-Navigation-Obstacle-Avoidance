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


COORDINATE_MATCHING_ACCURACY = 0.025


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

    def __init__(
        self,
        robot: Supervisor,
    ):

        self.robot = robot
        timestep = int(robot.getBasicTimeStep())

        robot.batterySensorEnable(timestep)

        receiver: Receiver = robot.getDevice("receiver1")
        receiver.enable(1)
        self.receiver = receiver

        emitter: Emitter = robot.getDevice("emitter1")
        self.emitter = emitter

        sensor_tags = [
            "ps6",
            "ps7",
            "ps0",
            "ps1",
        ]

        self.distance_sensors = []
        for sensor in sensor_tags:
            ps: DistanceSensor = robot.getDevice(sensor)
            ps.enable(1)
            self.distance_sensors.append(ps)

        self.control = robot_control.RobotControl(
            robot,
            speed=5,
        )
        self.started_at = self.robot.getTime()
        self.name = robot.getName()

        botCustomData = self.robot.getFromDef("BOT").getField("customData")

        self.client = BotClient(botCustomData)
        self.model = A2C.load("a2c-v3-30000")

    def check_battery(self):
        batteryLevel = self.robot.batterySensorGetValue()
        if batteryLevel <= 1:
            self.control.motor_stop()
            self.on_complete()

    def on_complete(self):
        self.control.motor_stop()
        batteryLevel = self.robot.batterySensorGetValue()
        energy_consumed = 1000 - batteryLevel

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
        weight = 0

        for sensor in self.distance_sensors:

            value = sensor.getValue()

            if value > 100:
                weight += 1

        if weight != 0:
            self.control.move(-0.7, 1)
            return True
        else:
            return False

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

    def predict(
        self,
        target_pos,
    ):
        avoidedObstacle = self.avoid_obstacles()
        if avoidedObstacle:
            return
        obs = np.asarray([self.get_alignment(target_pos)])

        action, _ = self.model.predict(obs)

        if np.average(action) > 0:
            self.control.move(action[0], action[1])
        else:
            self.control.move(1, 1)

    def run_hungarian(self, destinationCoordinates: list):
        if self.completed:
            return

        while True:

            x, y = self.control.x, self.control.y

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
