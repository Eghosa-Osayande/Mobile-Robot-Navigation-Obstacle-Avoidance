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

    def __init__(
        self,
        robot: Supervisor,
        speed=1
    ):
        super().__init__(
            id,
            [0,0],
            0,
        )
        self.speed = speed
        self.robot = robot

        left_wheel: Motor = robot.getDevice("left wheel motor")
        right_wheel: Motor = robot.getDevice("right wheel motor")
        left_wheel.setVelocity(0)
        right_wheel.setVelocity(0)
        left_wheel.setPosition(INF)
        right_wheel.setPosition(INF)
        self.left_wheel, self.right_wheel = left_wheel, right_wheel
        self.robot_step()

        gps: GPS = robot.getDevice("gps")
        gps.enable(1)
        self.gps = gps
        self.robot.step()

        compass: Compass = robot.getDevice("compass")
        compass.enable(1)
        self.compass = compass

        self.robot.step()

    def set_position_and_rotation(self):
        x, y, _ = self.gps.getValues()
        self.x=x
        self.y=y

        xo,yo,_=self.compass.getValues()
        oo=np.degrees(np.atan2(xo,yo))
        
        if oo<0:
            oo=360+oo
        if oo>=360:
            oo=oo-360
        self.orientation=oo

    def move(self, left_velocity, right_velocity):
        self.left_wheel.setVelocity(left_velocity * self.speed)
        self.right_wheel.setVelocity(right_velocity * self.speed)
        self.robot_step()
        self.set_position_and_rotation()

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
