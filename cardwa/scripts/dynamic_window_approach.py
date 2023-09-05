#! /usr/bin/env python
# -*- coding: utf-8 -*-

import rospy


from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose2D,PoseArray
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import LaserScan 
import math
from enum import Enum
import numpy as np


import time

import tf2_ros
from tf.transformations import euler_from_quaternion

class RobotType(Enum):
    circle = 0
    rectangle = 1
    
class DWA():
    def __init__(self):
        self.odom_subscriber = rospy.Subscriber("/odom", Odometry, self.odomCallback)
        self.scan_subscriber = rospy.Subscriber("/scan", LaserScan, self.scanCallback)

        # state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        self.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.obj_dis = 10000
        # goal position [x(m), y(m)]
        self.goal = np.array([10., 10.])

        # obstacles [x(m) y(m), ....]
        self.object_point = np.array([[-1, -1],
                   [0, 2],
                   [4.0, 2.0],
                   [5.0, 4.0],
                   [5.0, 5.0],
                   [6.0, 6.0],
                   [5.0, 9.0],
                   [8.0, 9.0],
                   [7.0, 8.0]
                   ])


        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.use_scan_data = False

    def getYaw(self, quat):
        q = [quat.x, quat.y, quat.z, quat.w]
        roll, pitch, yaw = euler_from_quaternion(q)

        return yaw


    def scanCallback(self, data):
        self.obj_dis = min(data.ranges)
        #print(np.array(data.ranges).size)
        
    def odomCallback(self, data):

        odom = data

        self.x[0] = odom.pose.pose.position.x
        self.x[1] = odom.pose.pose.position.y 
        self.x[2] = self.getYaw(odom.pose.pose.orientation)
        self.x[3] = odom.twist.twist.linear.x
        self.x[4] = odom.twist.twist.angular.z
        #print(self.x[2])

        
    def publishTwist(self, u):
        twist = Twist()
        twist.linear.x = u[0]
        twist.angular.z = u[1]

        self.pub.publish(twist)

    def dwa_control(self, x, config, goal, ob):
        """
        Dynamic Window Approach control
        """

        dw = self.calc_dynamic_window(x, config)

        u, trajectory = self.calc_control_and_trajectory(x, dw, config, goal, ob)

        return u, trajectory


    def calc_dynamic_window(self, x, config):
        """
        calculation dynamic window based on current state x
        """

        # Dynamic window from robot specification
        Vs = [config.min_speed, config.max_speed,
            -config.max_yawrate, config.max_yawrate]

        # Dynamic window from motion model
        Vd = [x[3] - config.max_accel * config.dt,
            x[3] + config.max_accel * config.dt,
            x[4] - config.max_dyawrate * config.dt,
            x[4] + config.max_dyawrate * config.dt]

        #  [vmin, vmax, yaw_rate min, yaw_rate max]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw

    def calc_control_and_trajectory(self, x, dw, config, goal, ob):
        """
        calculation final input with dynamic window
        """

        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        #print(dw[0], dw[1], dw[2], dw[3])

        # evaluate all trajectory with sampled input in dynamic window
        for v in np.arange(dw[0], dw[1], config.v_reso):
            for y in np.arange(dw[2], dw[3], config.yawrate_reso):
                trajectory = predict_trajectory(x_init, v, y, config)
                # calc cost
                to_goal_cost = config.to_goal_cost_gain * self.calc_to_goal_cost(trajectory, goal)
                speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
                ob_cost = config.obstacle_cost_gain * self.calc_obstacle_cost(trajectory, ob, config)
                #print( "ob",ob_cost)

                #print( "spped",speed_cost)

                #print( "togal",to_goal_cost)

                
                final_cost = to_goal_cost + speed_cost + ob_cost

                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, -y]
                    best_trajectory = trajectory
        #print("bu",best_u)        
        return best_u, best_trajectory



    def calc_obstacle_cost(self, trajectory, ob, config):
        """
            calc obstacle cost inf: collision
        """
        if self.use_scan_data:
            min_r = self.obj_dis
        else:
            ox = ob[:, 0]
            oy = ob[:, 1]
            dx = trajectory[:, 0] - ox[:, None]
            dy = trajectory[:, 1] - oy[:, None]
            r = np.hypot(dx, dy)-config.obstacle_R

            if config.robot_type == RobotType.rectangle:
                yaw = trajectory[:, 2]
                rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                rot = np.transpose(rot, [2, 0, 1])
                local_ob = ob[:, None] - trajectory[:, 0:2]
                local_ob = local_ob.reshape(-1, local_ob.shape[-1])
                local_ob = np.array([local_ob * x for x in rot])
                local_ob = local_ob.reshape(-1, local_ob.shape[-1])
                upper_check = local_ob[:, 0] <= config.robot_length / 2
                right_check = local_ob[:, 1] <= config.robot_width / 2
                bottom_check = local_ob[:, 0] >= -config.robot_length / 2
                left_check = local_ob[:, 1] >= -config.robot_width / 2
                if (np.logical_and(np.logical_and(upper_check, right_check),np.logical_and(bottom_check, left_check))).any():
                    return float("Inf")
            elif config.robot_type == RobotType.circle:
                if (r <= config.robot_radius).any():
                    return float("Inf")

            min_r = np.min(r)
            #print(min_r,self.obj_dis)
        return 1.0 / min_r  # OK


    def calc_to_goal_cost(self, trajectory, goal):
        """
            calc to goal cost with angle difference
        """

        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

        return cost

    def min_max_normalize(self, data):

        max_data = np.max(data)
        min_data = np.min(data)

        if max_data - min_data == 0:
            data = [0.0 for i in range(len(data))]
        else:
            data = (data - min_data) / (max_data - min_data)

        return data



def motion(x, u, dt):
    """
    motion model
    """


    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[2] += u[1] * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    """
    x = np.array(x_init)
    traj = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        traj = np.vstack((traj, x))
        time += config.dt

    return traj

