#! /usr/bin/env python
# -*- coding: utf-8 -*-

import rospy


from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose2D,PoseArray
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Quaternion



import math
from enum import Enum
import numpy as np


import tf2_ros
from tf.transformations import euler_from_quaternion

import dynamic_window_approach as dwa



class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 1.0  # [m/s]
        self.min_speed = 0  # [m/s]
        self.max_yawrate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_dyawrate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_reso = 0.01  # [m/s]
        self.yawrate_reso = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 1
        self.speed_cost_gain = 4
        self.obstacle_cost_gain = 0.5
        self.obstacle_R = 0.5
        self.robot_type = dwa.RobotType.circle
        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 0.20  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value



if __name__ == "__main__":
    rospy.init_node("dwa_ros")

    config = Config()
    config.robot_type = dwa.RobotType.circle

    dwa_ros = dwa.DWA()

    rospy.sleep(0.1)

    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        x = dwa_ros.x
        # print(x)
        ob = dwa_ros.object_point
        goal = dwa_ros.goal
        # print(goal)


        u, predicted_trajectory = dwa_ros.dwa_control(x, config, goal, ob)
        #print("p",predicted_trajectory)

        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= 0.5:
            u = [0.0, 0.0]
            dwa_ros.publishTwist(u)
        else:
            dwa_ros.publishTwist(u)

        rate.sleep()
    rospy.spin()


