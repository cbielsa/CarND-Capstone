#!/usr/bin/env python

class LowPassFilter(object):

    # Constructor
    def __init__(self, tau, ts):

        self.a = 1. / (tau / ts + 1.)
        self.b = tau / ts / (tau / ts + 1.);

        self.last_val = 0.
        self.ready = False


    # Reset filter
    def reset(self):

        self.last_val = 0.
        self.ready = False


    # Get filter state
    def get(self):

        return self.last_val


    # Pass value to filter
    def filt(self, val):
        
        if self.ready:
            val = self.a * val + self.b * self.last_val
        else:
            self.ready = True

        self.last_val = val

        return val
