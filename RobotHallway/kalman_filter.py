#!/usr/bin/env python3

# This is the code you need to do the Kalman filter for the door localization

import numpy as np

from robot_sensors import RobotSensors
from robot_ground_truth import RobotGroundTruth


# State estimation using Kalman filter
#   Stores the belief about the robot's location as a Gaussian
#     See the probability_sampling assignment, gaussian, for how to implement store probability as a Gaussian
# Slides for this assignment: https://docs.google.com/presentation/d/1Q6w-vczvWHanGDqbuz6H1qhTOkSrX54kf1g8NgTcipQ/edit?usp=sharing
class KalmanFilter:
    def __init__(self):
        # GUIDE: Store the Gaussian representing the location

        # YOUR CODE HERE

        # [position, velocity]
        self.x_mu = np.array([0,0])

        # Covariance matrix: position error and velocity error
        self.sigma = np.array([[0.4, 0],
                               [0, 0.1]])
        
        # top right should be a t for time delta, but we assume 1 time step
        self.A = np.array([[1, 0],
                           [0, 1]])
        
        # t squared/2 and t
        self.B = np.array([0.5, 0]).T

        self.C = np.array([1, 0])

        self.reset_kalman()

    def location_mean(self):
        """ Return the center of the gaussian (mu)
        @return mean : float"""
        # GUIDE Return your stored mu value for the Gaussian
        # YOUR CODE HERE

        return self.x_mu[0]
    
    def location_sigma(self):
        """ Return the sigma of the Gaussian
        @return sigma : float"""
        # GUIDE Return your stored sigma value for the Gaussian
        # YOUR CODE HERE

        return self.sigma[0][0]
    
    # Put robot in the middle with a really broad standard deviation
    def reset_kalman(self, loc: float = 0.5, sigma: float = 0.4):
        """ Reset location back to the middle of the hallway
        @param loc - the location in the unit interval
        @param signma - the sigma value for the Gaussian"""
        # GUIDE: Reset the location to the middle of the unit interval with a big sigma
        # YOUR CODE HERE

        self.x_mu[0] = loc
        self.x_mu[1] = 0.0  # initial velocity

        self.sigma[0][0] = sigma  # position variance
        self.sigma[0][1] = 0.0
        self.sigma[1][0] = 0.0
        self.sigma[1][1] = sigma  # velocity variance

        # print(f"mean: {self.x_mu}, sigma: {self.sigma}")

    # Sensor reading, distance to wall
    def update_belief_distance_sensor(self, robot_sensors: RobotSensors, dist_reading:float):
        """ Update state estimation based on sensor reading
        See Assignment slides for links to lecture slides
        @param robot_sensors - for mu/sigma of wall sensor
        @param dist_reading - distance reading returned by sensor
        @return None"""

        # GUIDE: Calculate C and K, then update the Gaussian
        # YOUR CODE HERE

        # print(self.C @ self.sigma @ self.C.T)

        # Kalman gain
        # K = self.sigma @ self.C.T @ np.linalg.inv(self.C @ self.sigma @ self.C.T + robot_sensors.wall_probs['sigma'])
        K = self.sigma @ self.C.T * 1/(self.C @ self.sigma @ self.C.T + robot_sensors.wall_probs['sigma'])

        # Update belief
        self.x_mu = self.x_mu + K * (dist_reading - self.C @ self.x_mu)
        self.sigma = (np.eye(2) - K @ self.C) @ self.sigma

    # Given a movement, update Gaussian
    def update_continuous_move(self, 
                               robot_ground_truth: RobotGroundTruth, 
                               amount: float):
        """ Kalman filter update mu/standard deviation with move (the prediction step)
        See Assignment slides for links to lecture slides
        @param robot_ground_truth : robot state - has the mu/sigma for moving
        @param amount : The control signal (the amount the robot was requested to move)
        @return : None """

        # GUIDE: Update mu and sigma by Ax + Bu equation
        # YOUR CODE HERE

        # Sensor noise
        R_sensor_sigma = np.array([[robot_ground_truth.move_probabilities["move_continuous"]["sigma"],0],
                                  [0,robot_ground_truth.move_probabilities["move_continuous"]["sigma"]]])

        # x_mu = A @ x_mu + B * amount
        self.x_mu = self.A @ self.x_mu + self.B * amount
        self.sigma = self.A @ self.sigma @ self.A.T + R_sensor_sigma

    def one_full_update(self, robot_ground_truth, robot_sensor, u: float, z: float):
        """This is the full update loop that takes in one action, followed by a sensor reading
        See Assignment slides for links to lecture slides
          Prediction followed by zensor reading
        Assumes the robot has been moved by the action u, then a sensor reading was taken (see test function below)
        @
        @param robot_sensor - has the robot sensor probabilities
        @param robot_ground_truth - robot location, has the probabilities for actually moving left if move_left called
        @param u will be the amount moved
        @param z will be the wall distance sensor reading
        """
        # GUIDE:
        #  Step 1 predict: update your belief by the action (move the Gaussian)
        #  Step 2 correct: do the correction step (move the Gaussian to be between the current mean and the sensor reading)
        # YOUR CODE HERE

        self.update_continuous_move(robot_ground_truth, u)
        self.update_belief_distance_sensor(robot_sensor, z)


if __name__ == '__main__':

    # Syntax checks
    kalman_filter_syntax = KalmanFilter()
    robot_ground_truth_syntax = RobotGroundTruth()
    robot_sensor_syntax = RobotSensors()

    # Set mu/sigmas
    sensor_noise_syntax = {"mu": 0.0, "sigma": 0.1}
    move_error_syntax = {"mu": 0.0, "sigma": 0.05}
    robot_ground_truth_syntax.set_move_continuos_probabilities(move_error_syntax["sigma"])
    robot_sensor_syntax.set_distance_wall_sensor_probabilities(sensor_noise_syntax["sigma"])

    kalman_filter_syntax.update_belief_distance_sensor(robot_sensor_syntax, 0.1)
    kalman_filter_syntax.update_continuous_move(robot_ground_truth_syntax, 0.0)

    # Do both at the same time
    u_syntax = 0.1
    robot_ground_truth_syntax.move_continuous(u_syntax)
    z_syntax = robot_sensor_syntax.query_distance_to_wall(robot_ground_truth_syntax)
    kalman_filter_syntax.one_full_update(robot_ground_truth_syntax, robot_sensor_syntax, u_syntax, z_syntax)

    # Generate some move sequences and compare to the correct answer
    from make_tests import test_kalman_update
    test_kalman_update()

    print("Done")
