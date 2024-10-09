# Design of a Navigation System for Mobile Robots in Unknown Terrain with Obstacle Avoidance

## Demo Video
https://github.com/user-attachments/assets/161f5343-562c-4861-8d31-9a1021e26b8d

## Abstract
This work presents an algorithmic framework for navigating mobile robots with location-based operational goals in unknown terrain, ensuring obstacle avoidance along the way. The core of the algorithm involves mapping the robot’s environment and continually updating this map based on the robot’s experiences at various locations. This real-time map is represented as a grid, where each cell corresponds to a specific area of the operating arena.

The navigation algorithm leverages the well-known A* algorithm, a widely-used pathfinding method that combines both actual travel cost and heuristic predictions to efficiently compute the optimal path. As the robot moves through the grid, it dynamically recalculates the best route whenever an obstacle is encountered. This allows the robot to adapt to unexpected changes in its environment.

By continuously updating the grid with information about obstacles and traversable areas, the robot builds an experience-based map that enhances its decision-making. This experience is stored in a data structure where grid cells are marked as visited, blocked, or available. The combination of real-time sensing and adaptive pathfinding ensures that the robot can successfully navigate to its target, even in unknown and dynamically changing environments.

## Methods

### Algorithm: A* Pathfinding

**Input:** 
- `start`: Starting position.
- `goal`: Goal position.
- `grid`: Map/grid where obstacles and traversable spaces are marked.
- `kind`: Movement type (4-way or 8-way).
- `limits`: Optional grid limits (default is infinite).

**Output:** 
- Path from start to goal (list of grid positions).

The A* (A-star) algorithm is widely used for efficient pathfinding and graph traversal, combining the strengths of Dijkstra's Algorithm and Greedy Best-First Search. It is heuristic-driven, meaning it estimates the cost from the current position to the goal and uses that knowledge to prioritize the next steps. The algorithm seeks to minimize the estimated total cost, `f(n) = g(n) + h(n)`, where:
- `g(n)` is the actual cost to reach a node `n`.
- `h(n)` is the heuristic estimate to the goal.

#### Heuristic Functions
- **Manhattan Distance:** Used for grid-based movement, it sums the absolute differences between current node coordinates and goal coordinates.
- **Euclidean Distance:** The straight-line distance between the current node and the goal, often used for 8-way movement.

#### Steps:
1. **Initialize Frontier:** A priority queue where the robot's start position is added with priority 0.
2. **Define Movement Types:**
   - 4-way movement: Up, down, left, right.
   - 8-way movement: Includes diagonals in addition to 4-way.
3. **Choose Heuristic:** 
   - For 4-way movement, Manhattan distance is used.
   - For 8-way movement, Euclidean distance is used.
4. **Path Calculation:** 
   - The algorithm explores neighboring nodes, updating the cost and re-evaluating the next steps until the goal is reached.
   - Obstacles in the grid (marked as `1`) are avoided.
5. **Path Reconstruction:** Once the goal is reached, the algorithm traces back from the goal to the start to reconstruct the optimal path.

### Algorithm: Move Robot to Target
1. **Initialize Robot Parameters:**
   - Define the robot's starting position, target, grid map, and scale factors.
2. **Convert Positions:** Real-world coordinates are converted into grid indices for use in A*.
3. **Calculate Initial Path:** The A* algorithm computes the path from the robot's current position to the goal.
4. **Subgoal Navigation:** The robot follows subgoals along the path.
   - Each subgoal is reached incrementally.
   - Obstacles are detected using sensors, and paths are recomputed when necessary.
5. **Obstacle Detection and Path Adjustment:** 
   - If obstacles block a path, the grid map is updated, marking the obstacle’s location, and a new path is computed.
6. **Robot Movement:** 
   - The robot rotates to face the next subgoal and moves forward unless an obstacle is detected.

### Robot Localization

Robot localization is key to determining where the robot is relative to its environment and its destination. For this project, a combination of GPS and compass sensors was used:
- **GPS**: Provides Cartesian coordinates in a global reference frame.
- **Compass**: Measures the robot's bearing to maintain orientation.

### Robot Locomotion

The robot uses a differential drive system, where the left and right wheels operate independently to enable forward, reverse, and turning motions:
- **Forward/Reverse:** Both wheels rotate at the same speed.
- **Turning:** Wheels rotate in opposite directions to initiate a turn.
- **Stop:** Both wheels halt to stop the robot.

### Obstacle Detection

Obstacle detection relies on sensors (camera, lidar, sonar, radar). For this work:
- **Forward and reverse obstacles** were detected using distance sensors.
- The robot focuses on objects along its direct path and recomputes the path when objects are detected within its range.

In our proposed system, an obstacle is defined as any object closer to the robot than the distance to the current target location.

### Path Planning

Path planning ensures the robot takes the most efficient path to the target while avoiding obstacles. The A* algorithm was chosen due to its ability to dynamically adjust paths based on real-time sensor data, recalculating when obstacles are detected.

The grid-based representation of the environment allows the robot to explore and record its path. When obstacles are encountered, they are marked, and the A* algorithm is used to find a new valid path to the target.

## Experiment

The proposed system was tested using an e-puck robot in Webots simulation software. The e-puck robot, equipped with a differential drive system, moves via independent control of its two wheels. The robot's movement types included:
- **Forward/Reverse Motion:** Using equal speeds for both wheels.
- **Turning:** By applying a phase difference between the wheel velocities.
- **Stopping:** By setting both wheel velocities to zero.

The e-puck robot was equipped with 8 distance sensors, labeled `ps0` to `ps7`. The forward sensors (`ps0` to `ps3`) were used to detect obstacles ahead, and the reverse sensors (`ps4` and `ps5`) detected obstacles behind the robot. A GPS and compass provided localization data.

### Simulation Details:
- **Arena:** 1x1 meter rectangular area.
- **Obstacles:** 20 cylindrical obstacles, each with a diameter of 0.06 meters, were randomly placed in five different scenarios.
- **Objective:** The robot navigated from a starting position to a predefined target while avoiding obstacles.

During navigation, when obstacles were detected, the robot stopped, updated its grid map, and recalculated its path using A*.

The results of the five different scenarios demonstrated the system’s ability to adapt to dynamic environments and successfully reach the target while avoiding obstacles.

