#!/usr/bin/env python

MIN_NUM = float('-inf')
MAX_NUM = float('inf')

class PID(object):

    # Constructor
    def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM):

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min = mn
        self.max = mx
        self.int_val = self.last_error = 0.


    # Reset integral value
    def reset(self):

        self.int_val = self.last_error = 0.


    def step(self, error, sample_time):

        # calculate state integral and derivative terms
        integral = self.int_val + error * sample_time
        derivative = (error - self.last_error) / sample_time

        # calculate actuation with PID control
        val = self.kp * error + self.ki * integral + self.kd * derivative

        # apply min and max allowed values
        val = max(self.min, min(val, self.max))

        # update controller state for next cycle
        self.int_val = integral
        self.last_error = error

        return val
