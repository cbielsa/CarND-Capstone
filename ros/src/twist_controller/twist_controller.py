#!/usr/bin/env python

import rospy
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter
import math

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):

	# Constructor
    def __init__(
        self,
    	sample_time,
    	decel_limit, vehicle_mass, wheel_radius, wheel_base, steer_ratio,
    	min_speed, max_lat_accel, max_steer_angle, fuel_capacity,
        brake_deadband ):

    	# attributes used to compute brake actuation
    	self.decel_limit = decel_limit
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius
        self.brake_deadband = brake_deadband

        self.brake_factor = abs(self.decel_limit)*self.wheel_radius*(
        	self.vehicle_mass + fuel_capacity*GAS_DENSITY )
        #rospy.loginfo('brake_factor : %f', self.brake_factor)

        # time of last cycle
        self.last_time = None

        # construct PID speed controller
        #kp = 0.5
        #ki = 0.0003
        #kd = 0.04
        kp = 0.5
        ki = 0.
        kd = 0.03

        # time between controller cycles if no latency [s]
        self.sample_time = sample_time
        self.speed_controller = PID(kp, ki, kd, -1., 1.)

        # construct yaw controller
        self.yaw_controller = YawController(
          	wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        # construct low pass filter for steering control
        #ts  = 1.
        #tau = 5.  # the larger tau, the more filtering (slower response to changes)
        #self.steer_filter = LowPassFilter(tau, ts)


    # Reset controller
    def reset(self):

    	# reset integral value of PID controller
    	self.speed_controller.reset()

    	# reset last time
    	self.last_time = None


    # Calculate actuations
    # inputs are expected to have been filtered by caller (if required)
    def control(self, v_target, w_target, v_current, w_current):

    	# calculate time elapsed since last cycle
    	time = rospy.get_time()
    	if self.last_time:
    		time_step = time - self.last_time
    	else:
    		time_step = self.sample_time
    	self.last_time = time

    	# calculate throttle/brake with PID controller (in range [-1, 1])
    	throttle = self.speed_controller.step(v_target - v_current, time_step)

    	# case brake
    	if throttle < 0:
    		brake = -throttle
    		throttle = 0.
    	else:
    		brake = 0.

        # avoid unnecessary braking
        # (apply brake control deadband)
        if brake < self.brake_deadband:
            brake = 0.

    	# convert brake from % to torque [Nm]
    	# torque = acc*wheel_radius*vehicle_mass
    	# TBC that what is commanded is total torque and not torque per wheel
    	if brake != 0.:
    		brake = brake*self.brake_factor

    	# calculate steering actuation with yaw controller [rad]

    	#w_target = self.w_target_filter.filt(w_target)
    	steer = self.yaw_controller.get_steering(v_target, w_target, v_current)

    	#rospy.loginfo(
    	#	'w_target: %f, w_current: %f, v_target: %f, v_current: %f, steer: %f deg',
    	#	w_target, w_current, v_target, v_current, math.degrees(steer))

        #rospy.loginfo(
        #    'v_target: %f, v_current: %f, dv: %f',
        #    v_target, v_current, v_target-v_current)

        #rospy.loginfo('throttle: %f, brake: %f, steer: %f',
        #    throttle, brake/self.brake_factor, steer)

        return throttle, brake, steer
