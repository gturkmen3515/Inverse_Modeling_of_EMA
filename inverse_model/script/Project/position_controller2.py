# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 13:46:23 2022

@author: gokmenatakanturkmen@gmail.com
"""

#!/usr/bin/env python3

import rospy
from dynamixel_workbench_msgs.srv import DynamixelCommand, DynamixelCommandRequest
from dynamixel_workbench_msgs.msg import DynamixelStateList
from std_msgs.msg import Float64MultiArray
import numpy as np

class DynamixelController:
    def __init__(self):
        rospy.init_node('dynamixel_workbench_server2')
        self.desired_value1 = 0
        self.desired_value2 = 0
        self.total_val = 0
        self.pos = []
        self.kk = DynamixelCommandRequest()
        self.kk.command = ''
        self.kk.id = 1

        rospy.wait_for_service('/dynamixel_workbench/dynamixel_command')
        self.service = rospy.ServiceProxy('/dynamixel_workbench/dynamixel_command', DynamixelCommand)

        rospy.Subscriber('/dynamixel_workbench/dynamixel_state', DynamixelStateList, self.callback)
        rospy.Subscriber('/linear', Float64MultiArray, self.callback2)

    def feed(self):
        rospy.spin()

    def callback(self, message):
        b1 = message.dynamixel_state[1]
        self.pos.append(b1.present_position)
        self.ema(float(self.pos[0]))

    def callback2(self, pos_message):
        self.desired_value1 = pos_message.data[0]
        self.desired_value2 = pos_message.data[1]
        print(self.desired_value1, self.desired_value2)

    def ema(self, pos):
        self.total_val = self.desired_value2
        self.total_val = np.clip(self.total_val, -20000, 20000)
        dyn_val = self.total_val + pos
        self.kk.addr_name = 'Goal_Position'
        self.kk.value = dyn_val
        result = self.service(self.kk)

if __name__ == '__main__':
    dynamixel_controller = DynamixelController()
    dynamixel_controller.feed()
