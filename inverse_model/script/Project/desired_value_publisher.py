# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 13:50:52 2022

@author: gokmenatakanturkmen@gmail.com
"""

#!/usr/bin/env python3

import rospy
import pandas as pd
from std_msgs.msg import Float64MultiArray

class DynamixelController:
    def __init__(self, excel_file_path='your_excel_file.xlsx'):
        # Load data from Excel file
        self.data = pd.read_excel(excel_file_path)
        self.index = 0

        # ... (other initialization code)

        # Create a publisher for the /linear topic
        self.linear_pub = rospy.Publisher('/linear', Float64MultiArray, queue_size=10)

    # ... (other methods)

    def get_next_data_point(self):
        if self.index < len(self.data):
            position = self.data.iloc[self.index]['Position']
            current = self.data.iloc[self.index]['Current']
            self.index += 1
            return position, current
        else:
            return None, None

    def publish_linear_topic(self):
        position, current = self.get_next_data_point()
        
        if position is not None and current is not None:
            # Create a Float64MultiArray message
            linear_msg = Float64MultiArray()
            linear_msg.data = [position, current]

            # Publish the message
            self.linear_pub.publish(linear_msg)
        else:
            rospy.loginfo("Reached the end of the data.")

if __name__ == '__main__':
    rospy.init_node('dynamixel_value_node')

    dynamixel_controller = DynamixelController()

    # Other initialization and code...

    # Publish on the /linear topic
    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        dynamixel_controller.publish_linear_topic()
        rate.sleep()
