#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):

    def __init__(self):

        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)
        min_speed = 0.1

        # Node state attributes
        self.v_target = 0.        # target linear velocity [m/s]
        self.w_target = 0.        # target angular velocity [rad/s]
        self.v_current = 0.       # current (measured) linear velocity [m/s]
        self.w_current = 0.       # current (measured) angular velocity [rad/s]
        self.dbw_enabled = False  # is DBW enabled

        self.controller_freq = 10  # controller frequency [Hz] 

        # Construct publishers
        self.steer_pub = rospy.Publisher(
            '/vehicle/steering_cmd', SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher(
            '/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher(
            '/vehicle/brake_cmd', BrakeCmd, queue_size=1)

        # Construct Controller object
        self.controller = Controller(
            1./self.controller_freq,
            decel_limit, vehicle_mass, wheel_radius, wheel_base, steer_ratio,
            min_speed, max_lat_accel, max_steer_angle)

        # Construct subscribers
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cmd_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)


        # Start control loop
        self.loop()


    # Callback function for /twist_cmd
    def twist_cmd_cb(self, twistStamped):

        # update target state
        self.v_target = twistStamped.twist.linear.x
        self.w_target = twistStamped.twist.angular.z
        
        #rospy.loginfo('Commanded v: %f, commanded w: %f', self.v_target, self.w_target)


    # Callback function for /current_velocity
    def current_velocity_cb(self, twistStamped):
        
        # update current (measured) state
        self.v_current = twistStamped.twist.linear.x
        self.w_current = twistStamped.twist.angular.z

        #rospy.loginfo('Measured v: %f, %f, %f, measured w: %f, %f, %f',
        #    self.v_current,
        #    twistStamped.twist.linear.y,
        #    twistStamped.twist.linear.z,
        #    twistStamped.twist.angular.x,
        #    twistStamped.twist.angular.y,
        #    self.w_current
        #    )


    # Callback function for /vehicle/dbw_enabled
    def dbw_enabled_cb(self, enabled):

        if not self.dbw_enabled and enabled:
            
            # set DBW status to enabled
            self.dbw_enabled = True
            rospy.loginfo('DBW enabled')

        elif self.dbw_enabled and not enabled:
            
            # reset integral state of PID speed controller
            self.controller.reset()

            # set DBW status to disabled
            self.dbw_enabled = False
            rospy.loginfo('DBW disabled')

    
    def loop(self):

        # [cbielsa]: reduced rate due to limitations in my system
        rate = rospy.Rate(10) # 50Hz

        while not rospy.is_shutdown():

            # TODO: Get predicted throttle, brake, and steering using `twist_controller`
            # You should only publish the control commands if dbw is enabled
            # throttle, brake, steering = self.controller.control(<proposed linear velocity>,
            #                                                     <proposed angular velocity>,
            #                                                     <current linear velocity>,
            #                                                     <dbw status>,
            #                                                     <any other argument you need>)
            

            if self.dbw_enabled:

                #rospy.loginfo('dbw enabled')

                # call controller and publish actuations
                throttle, brake, steer = self.controller.control(
                    self.v_target, self.w_target, self.v_current, self.w_current)

                self.publish(throttle, brake, steer)


            # sleep until next control cycle
            rate.sleep()


    def publish(self, throttle, brake, steer):

        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
