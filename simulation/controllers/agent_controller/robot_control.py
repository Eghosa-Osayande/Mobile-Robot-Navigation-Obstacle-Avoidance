"""agent controller."""

import math
import numpy as np

from controller import (
    Motor,
    Receiver,
    GPS,
    Compass,
    Emitter,
    Supervisor,
)


ROBOT_ROTATIONAL_SPEED = 2
ROBOT_TANGENTIAL_SPEED = 5.0
OBSTACLE_DETECTION_THRESHOLD = 120
INF = float("inf")


class RobotBase:

    position: list[float]
    orientation: float = 0
    obstacle_threshold = 1
    id: int

    def __init__(
        self,
        id: int,
        initial_position,
        orientation=0,
    ):
        self.id = id
        self.position = initial_position
        self.orientation = orientation

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @x.setter
    def x(self, v):
        self.position[0] = v

    @y.setter
    def y(self, v):
        self.position[1] = v

    def move(self, left_velocity, right_velocity):
        pass

    def is_out_of_range(self):
        pass


class RobotControl(RobotBase):

    robot: Supervisor
    left_wheel: Motor
    right_wheel: Motor

    gps: GPS
    compass: Compass
    isReversing:bool=False

    def __init__(self, robot: Supervisor, speed=1):
        super().__init__(
            id,
            [0, 0],
            0,
        )
        self.speed = speed
        self.robot = robot
        self.isReversing=False

        left_wheel: Motor = robot.getDevice("left wheel motor")
        right_wheel: Motor = robot.getDevice("right wheel motor")
        left_wheel.setVelocity(0)
        right_wheel.setVelocity(0)
        left_wheel.setPosition(INF)
        right_wheel.setPosition(INF)
        self.left_wheel, self.right_wheel = left_wheel, right_wheel
        self.robot_step()

        gps: GPS = robot.getDevice("gps")
        self.gps = gps

        compass: Compass = robot.getDevice("compass")
        self.compass = compass

        self.robot.step()

    def sync(self):
        self.gps.enable(1)
        self.robot.step()
        x, y, _ = self.gps.getValues()
        self.gps.disable()

        self.x = x
        self.y = y

        self.compass.enable(1)
        self.robot.step()
        xo, yo, _ = self.compass.getValues()
        self.compass.disable()
        oo = np.degrees(np.atan2(xo, yo))

        if oo < 0:
            oo = 360 + oo
        if oo >= 360:
            oo = oo - 360

        self.orientation = oo

    def move(self, left_velocity, right_velocity):
        max_v = 6.28
        self.left_wheel.setVelocity(np.clip(left_velocity * self.speed, -max_v, max_v))
        self.right_wheel.setVelocity(
            np.clip(right_velocity * self.speed, -max_v, max_v)
        )
        self.isReversing=(left_velocity,right_velocity)==(-1,-1)
        self.robot_step()

    def motor_rotate_left(self):
        self.move(1, 0)

    def motor_rotate_right(self):
        self.move(0, 1)

    def motor_move_forward(self):
        self.move(1, 1)

    def motor_stop(self):
        self.move(0, 0)

    def robot_step(self, timestep=None):
        self.robot.step(timestep)

    def detect_obstacles(self):
       

        (0.03, 234.93, 0.0241),
        (0.04, 158.03, 0.0287),
        (0.05, 120, 0.04225),
        (0.06, 104.09, 0.03065),
        (0.07, 67.19, 0.04897),

        thresholds = {
            "ps6": 250,
            "ps7": 100,
            "ps0": 100,
            "ps1": 250,
            "ps3":100, 
            "ps4":100,
        }

        values = {}
        actualValues = []

        for tag, thres in thresholds.items():
            ps = self.robot.getDevice(tag)
            ps.enable(1)
            self.robot.step()
            value = ps.getValue()
            ps.disable()
            actualValues.append(value)
            values[tag] = 1 if value >= thres else 0
        
        detected_obs = [0,0]

        # front
        if values["ps7"] or values["ps0"] or (values["ps6"] and values["ps1"]):
            detected_obs[0] = 1
        # back
        if values["ps3"] or values["ps4"]:
            detected_obs[1] = 1
        
        return detected_obs
