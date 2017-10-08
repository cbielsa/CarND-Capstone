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
    def __init__(self,
    	sample_time,
    	decel_limit, vehicle_mass, wheel_radius, wheel_base, steer_ratio,
    	min_speed, max_lat_accel, max_steer_angle):

    	# attributes used to compute brake torque
    	self.decel_limit = decel_limit
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius

        # construct PID speed controller
        kp = 0.2
        ki = 0.
        kd = 0.
        self.sample_time = sample_time  # time between controller cycles [s]
        self.speed_controller = PID(kp, ki, kd, -1., 1.)

        # construct yaw controller
        self.yaw_controller = YawController(
          	wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        # construct low pass filter for steering control
        #ts  = 1.
        #tau = 3.  # the larger tau, the more filtering (slower response to changes)
        #self.steer_filter = LowPassFilter(tau, ts)
        #self.w_target_filter = LowPassFilter(tau, ts)


    # Reset controller
    def reset(self):

    	# reset integral value of PID controller
    	self.speed_controller.reset() 


    # Calculate actuations
    def control(self, v_target, w_target, v_current, w_current):

    	rospy.loginfo('v_target: %f, v_current: %f', v_target, v_current)
        
    	# calculate throttle/brake with PID controller (in range [-1, 1])
    	throttle = self.speed_controller.step(v_target - v_current, self.sample_time)

    	# case brake
    	if throttle < 0:
    		brake = -throttle
    		throttle = 0.
    	else:
    		brake = 0.

    	# convert brake from % to torque [Nm]
    	# torque = acc*wheel_radius*vehicle_mass
    	# TBC that what is commanded is total torque and not torque per wheel
    	if brake != 0.:
    		brake = brake*abs(self.decel_limit)*self.wheel_radius*self.vehicle_mass

    	# calculate steering actuation with yaw controller [rad]

    	#w_target = self.w_target_filter.filt(w_target)
    	steer = self.yaw_controller.get_steering(v_target, w_target, v_current)

    	# filter steering actuation
    	#steer = self.steer_filter.filt(steer)

    	rospy.loginfo('w_target: %f, v_target: %f, v_current: %f, steer: %f deg',
    		w_target, v_target, v_current, math.degrees(steer))

        return throttle, brake, steer
