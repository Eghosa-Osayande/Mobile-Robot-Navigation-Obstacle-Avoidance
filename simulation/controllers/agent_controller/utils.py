import math
import numpy as np
from controller import Supervisor
import heapq


def is_coordinate_equal(coordinate1, coordinate2, matching_accuracy=0):
    """
    Check if two coordinates are equal within a specified accuracy and return the position error.

    Parameters:
    - coordinate1: tuple or list, first (x, y) coordinate
    - coordinate2: tuple or list, second (x, y) coordinate
    - matching_accuracy: float, the maximum allowable difference for the coordinates to be considered equal

    Returns:
    - is_equal: bool, True if the coordinates are equal within the specified accuracy, False otherwise
    - position_error: tuple, the absolute difference in the x and y coordinates
    """

    is_equal = (
        math.fabs(coordinate1[0] - coordinate2[0]) < matching_accuracy
        and math.fabs(coordinate1[1] - coordinate2[1]) < matching_accuracy
    )

    position_error = math.sqrt(
        (coordinate1[0] - coordinate2[0]) ** 2 + (coordinate1[1] - coordinate2[1]) ** 2
    )

    return is_equal, position_error


def calc_destination_theta_in_degrees(current_coordinate, destination_coordinate):
    theta = math.atan2(
        destination_coordinate[1] - current_coordinate[1],
        destination_coordinate[0] - current_coordinate[0],
    )
    thetaDeg = theta * 180 / math.pi

    return thetaDeg


def calc_theta_dot(heading, destination_theta):
    theta_dot = destination_theta - heading

    if theta_dot > 180:
        theta_dot = -(360 - theta_dot)
    elif theta_dot < -180:
        theta_dot = 360 + theta_dot

    return theta_dot


import math


def calc_distance(current_coordinate, destination_coordinate):
    return math.sqrt(
        math.pow(destination_coordinate[0] - current_coordinate[0], 2)
        + math.pow(destination_coordinate[1] - current_coordinate[1], 2)
    )


def angles_to_rotation(phi, theta, psi):

    angs = np.radians((phi, theta, psi))

    maxang = np.max(np.abs(angs))

    if maxang == 0:

        return [0, 0, 1, 0]

    signs = np.array(list(-1 if ang < 0 else +1 for ang in angs))

    fracs = np.sqrt(np.abs(angs) / maxang)

    rs = signs * fracs

    return [rs[0], rs[1], rs[2], maxang]


def anticlockwise_difference(start, end):
    # Normalize both angles to be within -180 to 180 degrees
    start = (start + 180) % 360 - 180
    end = (end + 180) % 360 - 180

    # Calculate the direct difference
    difference = end - start

    # Adjust difference for anticlockwise motion if necessary
    if difference < 0:
        difference += 360

    return difference


def show_target(robot: Supervisor, destinationCoordinates):
    root_node = robot.getRoot()
    children_field = root_node.getField("children")
    children_field.importMFNodeFromString(
        -1,
        f"""
            TargetProto {{
                translation {destinationCoordinates[0]} {destinationCoordinates[1]} 0.1
            }}
        """,
    )
    robot.step()


def insert_obstacle(robot: Supervisor, destinationCoordinates):
    root_node = robot.getRoot()
    children_field = root_node.getField("children")
    children_field.importMFNodeFromString(
        -1,
        f"""
            DEF epuck E-puck {{
                translation {destinationCoordinates[0]} {destinationCoordinates[1]} 0
                controller "<none>"
            }}
        """,
    )
    robot.step()


def map_sensor_value(sensor_value):
    # Define the lookup table as a list of tuples (sensor value, sensor reading, real value)
    lookup_table = [
        (0, 4095, 0.002),
        (0.005, 2133.33, 0.003),
        (0.01, 1465.73, 0.007),
        (0.015, 601.46, 0.0406),
        (0.02, 383.84, 0.01472),
        (0.03, 234.93, 0.0241),
        (0.04, 158.03, 0.0287),
        (0.05, 120, 0.04225),
        (0.06, 104.09, 0.03065),
        (0.07, 67.19, 0.04897),
    ]

    # If the sensor value is exactly one of the known values, return the corresponding real value
    for i in range(len(lookup_table)):
        if sensor_value == lookup_table[i][1]:
            return lookup_table[i][2]

    # Otherwise, perform linear interpolation between the closest known points
    for i in range(len(lookup_table) - 1):
        if lookup_table[i][1] >= sensor_value >= lookup_table[i + 1][1]:
            # Linear interpolation formula
            x0, y0 = lookup_table[i][1], lookup_table[i][2]
            x1, y1 = lookup_table[i + 1][1], lookup_table[i + 1][2]
            return y0 + (sensor_value - x0) * (y1 - y0) / (x1 - x0)

    # If the sensor value is outside the known range, return None or handle it as needed
    print("wtf", sensor_value)
    return 0


def calculate_distance(point1, point2):
    # Convert points to NumPy arrays
    point1 = np.array(point1)
    point2 = np.array(point2)

    # Calculate the distance
    distance = np.linalg.norm(point1 - point2)

    return np.abs(distance)


def get_alignment_to_target(orientation, current_coordinate, destination_coordinate):
    theta = np.arctan2(
        destination_coordinate[1] - current_coordinate[1],
        destination_coordinate[0] - current_coordinate[0],
    )

    return np.degrees(theta) - orientation


def point_to_index(cord, offset=0, scale=1):
    x, y = cord

    pos = (int((x + offset) // scale), int((y + offset) // scale))

    remainders = ((x + offset) % scale, (y + offset) % scale)
    return pos, remainders


def index_to_point(cord, offset=0, scale=1, remainders=(0, 0)):
    xx = (cord[0] * scale) - offset + remainders[0]
    yy = (cord[1] * scale) - offset + remainders[1]
    return xx, yy


def generate_sub_moves(start, target):

    current_position = list(start)
    target_position = list(target)

    moves = []

    while current_position != target_position:
        move = (0, 0)  # No movement initially

        # Check horizontal movement
        if current_position[0] < target_position[0]:
            move = (1, 0)  # Move right
        elif current_position[0] > target_position[0]:
            move = (-1, 0)  # Move left

        # Check vertical movement
        if current_position[1] < target_position[1]:
            move = (move[0], 1)  # Move up
        elif current_position[1] > target_position[1]:
            move = (move[0], -1)  # Move down

        current_position[0] += move[0]
        current_position[1] += move[1]

        moves.append(tuple(current_position))

    return moves


# A* Algorithm
def a_star(
    start,
    goal,
    grid,
    kind,
    limits=[(-float("inf"), float("inf")), (-float("inf"), float("inf"))],
):

    def heuristic1(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def heuristic2(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(pos, neighbors):
        valid_neighbors = []
        for dx, dy in neighbors:
            newx, newy = pos[0] + dx, pos[1] + dy

            if (
                limits[0][0] <= newx < limits[0][1]
                and limits[1][0] <= newy < limits[1][1]
                and grid.get((newx, newy)) != 1
            ):
                valid_neighbors.append((newx, newy))
        return valid_neighbors

    # Initialization
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    # Algorithm loop
    path = []
    neighbors = {
        heuristic2: [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
        ],  # 4-way movement
        heuristic1: [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
        ],
    }

    movement_kind = {
        4: heuristic2,  # 4-way movement
        8: heuristic1,
    }

    heuristic = movement_kind[kind]

    while frontier:
        current_cost, current = heapq.heappop(frontier)

        if current == goal:
            break

        for next in get_neighbors(current, neighbors[heuristic]):
            new_cost = cost_so_far[current] + 1  # Assuming uniform cost
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current

    # Reconstruct path
    current = goal

    while current != start:
        path.append(current)
        current = came_from.get(current)
        if current is None:
            path = []
            break

    if len(path) != 0:
        path.append(start)
        path.reverse()

    return path
