#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32 
from geometry_msgs.msg import PoseStamped, TwistStamped, Point
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray, TrafficLight, TrafficLightStateAndWP

import math
import tf
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
'''


# Number of waypoints we will publish. You can change this number
LOOKAHEAD_WPS = 150

MAX_LOOKAHEAD_WPS = 40
MAX_LOOKAHEAD_DIST = 40. #mts

# Max allowed speed (limit set by waypoint_loader)
MAX_SPEED  = rospy.get_param('/waypoint_loader/velocity') *1000/3600  # m/s
rospy.loginfo('MAX_SPEED: %f m/s', MAX_SPEED)

MAX_ACC    = 9.  # m/s^2
BRAKE_DECC = 2.  # nominal decceleration used to stop car [m/s^2]
                 # (ego will be braked at up to MAX_ACC if light suddently turns red)


# due to PID characteristics the braking may overshoot target
# Take that into account
MARGIN_FOR_BRAKE_OVERSHOOT = 5.  

# Braking distance when ego is at maximum speed and brakes at nominal decceleration
MAX_BRAKE_DIST = MAX_SPEED*MAX_SPEED/(2.*BRAKE_DECC) + MARGIN_FOR_BRAKE_OVERSHOOT

# Used to compensate for the delays in system latencies when publishing waypoints
TIME_COMPENSATE_PROP_DELAYS = 0.3 # in seconds

#Length of the car
CAR_LENGTH = 6. #mts

#Distance from stopline to actual Traffic light
STOP_LINE_TO_TL_DIST = 20.

class WaypointUpdater(object):

    def __init__(self):

        # Initialize node
        rospy.init_node('waypoint_updater')

        # Construct subscribers
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', TrafficLightStateAndWP, self.traffic_cb)
        #rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb)

        # Construct publisher
        self.final_waypoints_pub = rospy.Publisher(
            '/final_waypoints', Lane, queue_size=1)

        # other member variables
        self.base_waypoints = None
        self.final_waypoints = None
        self.base_waypoints_s = [] # base_waypoints segment numbers
        self.total_waypoints = 0
        
        # index of closest waypoint to stop line or traffic light,
        # depending on message passed by '/traffic_waypoint'
        # (set to none if no red light in front of car)
        self.tl_cur_wp_ix = 0
        self.tl_cur_state = TrafficLight.UNKNOWN
        self.tl_cur_state_first_detect_time = None
        
        self.tl_prev_state = TrafficLight.UNKNOWN
        self.tl_prev_wp_ix = 0
        self.tl_prev_change_time = None

        self.tl_red_duration       = 10. #secs. UNUSED
        self.tl_green_duration     = 3.  #secs. UNUSED
        self.tl_yellow_duration    = 2.  #secs. UNUSED
        
        # set to true if message '/traffic_waypoint' contains closest wp to traffic light
        # set to false if message '/traffic_waypoint' contains closest wp to stop line
        self.correct_traffic_wp_to_stopline = False
        
        # if self.correct_traffic_wp_to_stopline = True,
        # this is the distance [m] between a stop line and its traffic light
        self.margin_to_tl = 30.

        # current velocity
        self.current_velocity = None

        # last measured stamped pose
        self.meas_pose = None

        # waypoint updater publishing frequency
        self.freq = 2  # Hz

        # flag set to truth first time an image is received
        self.traffic_received = False

        rospy.loginfo("")
        rospy.loginfo("              Max speed: %0.2f mts/sec", MAX_SPEED)
        rospy.loginfo("               Max accl: %0.2f mts/sec^2", MAX_ACC)
        rospy.loginfo("           Brake deccel: %0.2f mts/sec^2", BRAKE_DECC)
        rospy.loginfo(" Brake overshoot margin: %0.2f mts", MARGIN_FOR_BRAKE_OVERSHOOT)
        rospy.loginfo("   Max braking Distance: %0.2f mts", MAX_BRAKE_DIST)
        rospy.loginfo("Max lookahead waypoints: %d", MAX_LOOKAHEAD_WPS)
        rospy.loginfo("System delay to pub wps: %0.2f", TIME_COMPENSATE_PROP_DELAYS)
        rospy.loginfo("")
        
        # Start waypoint updater loop
        self.loop()


    # Auxiliary functions ============================================

    # Return True if updater has received all messages needed
    # to calculate final waypoints
    def initialized(self):

        if( self.base_waypoints and self.current_velocity
            and self.meas_pose and self.traffic_received and ( rospy.get_rostime()!=0 ) ):
            return True
        else:
            return False

    def get_ego_wp_at_t(self, t_sec, vel, accl):

        # get current time (class rospy.Time)
        cur_t = rospy.get_rostime()

        # Add time t_sec
        to_add = rospy.Duration(t_sec) 

        # New time in seconds
        new_t = cur_t + to_add
        
        # get last measured stamped pose
        prev_pos = self.meas_pose.pose.position

        # the previous position that we received for ego
        prev_heading = self.get_heading_from_quaternion(self.meas_pose.pose.orientation)
        prev_position = self.meas_pose.pose.position
                
        # set init waypoint index to waypoint in front of ego
        prev_wp = self.next_waypoint(prev_position, prev_heading)
        
        # Calculate dt
        dt = ( new_t - self.meas_pose.header.stamp ).to_sec()

        # Calculate the distance ego vehicle would be travelling in dt at current velocity
        dist = vel*dt + accl*dt*dt

        # Calculate the wp the car will be at
        pred_wp = self.find_wp_at_distance_after(prev_wp, dist)

        return pred_wp, dist
        
        
    # Predict ego position at given input time,
    # propagating with last available pose and velocity measurements
    # Last measured heading is also returned
    #
    # If time==Null, last pose is returned, w/o propagation
    def get_pose(self, time):

        # get last measured stamped pose
        pos = self.meas_pose.pose.position

        # calculate ego heading from last measured pose
        heading = self.get_heading_from_quaternion(self.meas_pose.pose.orientation)

        if time:
            # predict position at input time, using last measured velocity
            # and assuming constant heading
            dt = ( time - self.meas_pose.header.stamp ).to_sec()
            pos.x += self.current_velocity*math.cos(heading)*dt
            pos.y += self.current_velocity*math.sin(heading)*dt

        return pos, heading  # pos in [m], heading in [rad]


    # Return index of closest waypoint
    # self.base_waypoints shall be defined 
    def closest_waypoint(self, ego_position):

        min_dist2 = 1e12
        wp_index = 0  # index of closest waypoint

        for index, waypoint in enumerate( self.base_waypoints.waypoints ):

            dist2 = self.dist(
                waypoint.pose.pose.position,
                ego_position)

            if dist2 < min_dist2:
                wp_index = index
                min_dist2 = dist2

        return wp_index


    def get_heading_from_quaternion(self, quaternion):

        # compute euler angles from quaternion
        euler = tf.transformations.euler_from_quaternion(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w])

        return euler[2]


    def is_behind_ego(self, ego_heading, other_heading):

        angle = abs(ego_heading - other_heading)
        if angle > math.pi:
            angle = abs(2.*math.pi - angle)

        if angle > math.pi/2.:
            return True
        
        return False
    

    # Return index of next waypoint
    # self.base_waypoints shall be defined 
    def next_waypoint(self, ego_position, ego_heading):

        ego_x = ego_position.x
        ego_y = ego_position.y
        
        # find closest waypoint
        closest_wp_index = self.closest_waypoint(ego_position)
        closest_wp = self.base_waypoints.waypoints[closest_wp_index]

        # calculate the heading from ego to closest waypoint, in global frame [rad]
        closest_wp_x = closest_wp.pose.pose.position.x
        closest_wp_y = closest_wp.pose.pose.position.y
        closest_wp_heading = math.atan2(closest_wp_y-ego_y, closest_wp_x-ego_x)

        # if closest waypoint is behind ego, take next wp
        if self.is_behind_ego(ego_heading, closest_wp_heading):
            closest_wp_index += 1

        closest_wp_index = closest_wp_index%self.total_waypoints
            
        return closest_wp_index


    # Subscriber callback functions ==================================

    # Callback function for /current_pose
    def pose_cb(self, poseStamped):
        
        self.meas_pose = poseStamped

    # Callback function for /current_velocity
    def velocity_cb(self, twistStamped):
        
        # update current (measured) state
        self.current_velocity = twistStamped.twist.linear.x

    # Callback function for /base_waypoints
    #
    # For info: in SIM, there are 10902 wps at an average distance of 0.64 m,
    # but distance between consecutive wps are actually quite variable
    # Hence waypoint index cannot be used as a reliable proxy of distance
    def waypoints_cb(self, waypoints):

        if not self.base_waypoints:

            #rospy.loginfo('Copying base_waypoints...')
            self.base_waypoints = waypoints

            self.total_waypoints = len(waypoints.waypoints)
            
            rospy.loginfo("Total number of waypoints = %d", self.total_waypoints)
            
            self.base_waypoints_s.append(0)
            
            rospy.loginfo('Computing base_waypoint segments...')
            for ix in range(len(waypoints.waypoints)-1):
                self.base_waypoints_s.append(self.dist_ix(ix, ix+1))
        
            for ix in range(len(self.base_waypoints_s)-1):
                self.base_waypoints_s[ix+1] += self.base_waypoints_s[ix]


    def tl_get_color_str(self, state):
        state_str = "UNKNOWN"
        
        if(state == TrafficLight.RED):
            state_str = "RED"
        elif(state == TrafficLight.GREEN):
            state_str = "GREEN"
        elif(state == TrafficLight.YELLOW):
            state_str = "YELLOW"

        return state_str
    
    # Callback function for /traffic_waypoint message 
    def traffic_cb(self, msg):

            
        state = msg.state
        wp_ix = msg.wp_ix
        
        if((state != self.tl_cur_state)
           or (wp_ix != self.tl_cur_wp_ix)):
            self.tl_prev_state = self.tl_cur_state
            self.tl_prev_wp_ix = self.tl_cur_wp_ix
            self.tl_cur_state = state
            self.tl_cur_wp_ix = wp_ix
            self.tl_cur_state_first_detect_time = msg.first_detect_time
            
            rospy.loginfo("WP TL Notif: change %s(%d)->%s(%d)",
                          self.tl_get_color_str(self.tl_prev_state),
                          self.tl_prev_wp_ix,
                          self.tl_get_color_str(self.tl_cur_state),
                          self.tl_prev_wp_ix)

            #rospy.loginfo("  TL NOTIF: first detect of %s at %0.2f",
            #              self.tl_get_color_str(self.tl_cur_state),
            #              self.tl_cur_state_first_detect_time.to_sec())
            
            
            # get current time (class rospy.Time)
            curr_time = rospy.get_rostime()

            if(self.tl_prev_state == TrafficLight.RED):
                self.tl_red_duration = (curr_time - self.tl_prev_change_time).to_sec()
            if(self.tl_prev_state == TrafficLight.GREEN):
                self.tl_green_duration = (curr_time - self.tl_prev_change_time).to_sec()
            if(self.tl_prev_state == TrafficLight.YELLOW):
                self.tl_yellow_duration = (curr_time - self.tl_prev_change_time).to_sec()

            #rospy.loginfo("  TL NOTIF: Durations: Red:%d secs, Green:%d secs, Yellow:%d secs",
            #              self.tl_red_duration, self.tl_green_duration, self.tl_yellow_duration)

            self.tl_prev_change_time = msg.header.stamp

            
        # set availability flag
        self.traffic_received = True

    # ix2 is expected to be greater than ix1
    def dist_between_waypoints(self, ix1, ix2):
        num_waypoints = abs(ix2 - ix1)
        if(num_waypoints > (self.total_waypoints/2)):
            next_wp = ix1
            num_waypoints = (self.total_waypoints - ix1) + ix2

            # wrapped around. walk through and get distance
            rospy.loginfo(" ")
            rospy.loginfo(" WRAPPED AROUND! (ix1=%d, ix2=%d). num_wps %d", ix1, ix2, num_waypoints)
            rospy.loginfo(" ")

            # Approximation...
            tot_dist = self.dist(self.base_waypoints.waypoints[ix1].pose.pose.position,
                                 self.base_waypoints.waypoints[ix2].pose.pose.position)
            
            #for i in range(num_waypoints):
            #    tot_dist += self.dist_to_very_next_waypoint(ix1, next_wp)
            #    ix1 = next_wp
            #    next_wp = self.find_very_next_wp_after(ix1)

            return tot_dist
        
        else:
            return (self.base_waypoints_s[ix2] - self.base_waypoints_s[ix1])

        
    # ix2 is expected to be greater than ix1
    def dist_to_very_next_waypoint(self, ix1, ix2):
        ix2 = ix2 % self.total_waypoints
        if(ix1 < 0):
            ix1 = self.total_waypoints + ix1

        if(ix2 < ix1):
            rospy.loginfo(" ")
            rospy.loginfo("ONLY possible at wraparound (ix1=%d, ix2%d)", ix1, ix2)
            rospy.loginfo(" ")
            # Approximation
            return self.dist(self.base_waypoints.waypoints[ix1].pose.pose.position,
                             self.base_waypoints.waypoints[ix2].pose.pose.position)
        else:
            return (self.base_waypoints_s[ix2] - self.base_waypoints_s[ix1])


    def find_very_next_wp_after(self, wp_ix):
        return (wp_ix + 1)%self.total_waypoints
    
    def find_wp_at_distance_after(self, wp_ix, d):
        next_wp = wp_ix
        tot_dist = 0
        
        while(True):
            tot_dist += self.dist_to_very_next_waypoint(wp_ix, next_wp)
            if(tot_dist >= d):
                break
            else:
                wp_ix = next_wp
                next_wp = self.find_very_next_wp_after(wp_ix)

        return next_wp
    
    def ego_obeyed_prev_tl(self, ego_next_wp_ix):
        if(self.tl_prev_wp_ix == None):
            return True
        else:
            if(ego_next_wp_ix > self.tl_prev_wp_ix+20):
                return True
            else:
                if(self.current_velocity < 0.5):
                    return True
                else:
                    return False

    # Due to system latencies we must predict where ego will be
    # taking propogation delays into account.
    # Based on the above, we decide how to respond to the light
    def get_light_state_to_use(self, ego_wp, tl_stopline_wp):
        cur_state = self.tl_cur_state

        accelerate = False

        dist_to_stopline = self.dist_between_waypoints(ego_wp, tl_stopline_wp)
        
        # We have to take into account actuation delays to predict where ego will be when
        # the actualt actuation will go through. Just using some multiples that worked.
        # Also, ideally a way to know how long a light will remain green  would be good.
        # But this information will be different for different lights (e.g. simulation vs.
        # site) so we cant make simple assumptions +  we dont have a way to get this info.
        
        cur_time = rospy.get_rostime()

        tl_on_for_secs = (cur_time - self.tl_cur_state_first_detect_time).to_sec()
        
        rospy.loginfo("WP TL:%s: stopline_wp=%d, on_for=%0.2f secs, ego_wp=%d, ",
                      self.tl_get_color_str(self.tl_cur_state),
                      tl_stopline_wp, tl_on_for_secs, ego_wp)

        rospy.loginfo("      current_velocity = %0.2f", self.current_velocity)


        if(self.tl_cur_state == TrafficLight.GREEN):
            # Predict where will ego be in the future

            # Would we have crossed the light?

            secs = 4*TIME_COMPENSATE_PROP_DELAYS #seconds
            wp_in_secs, dist_covered = self.get_ego_wp_at_t(secs,
                                                            self.current_velocity, MAX_ACC)

            #tl_wp = tl_stopline_wp + (STOP_LINE_TO_TL_DIST/2.)

            dist_to_stop = dist_to_stopline + STOP_LINE_TO_TL_DIST/2. - (CAR_LENGTH/2.)
            
            if((dist_covered < dist_to_stop) and (tl_on_for_secs > 3)):
                cur_state = TrafficLight.RED
                rospy.loginfo("      Cannot cross tl xsection. Dist to stop %0.2f. Dist covered %0.2f.Prepare to stop...",
                              dist_to_stop, dist_covered)
            else:
                rospy.loginfo("      Can cross tl xsection. Dist to stop %0.2f. Dist covered %0.2f. Continue on...",
                              dist_to_stop, dist_covered)
                accelerate = True

        elif(self.tl_cur_state == TrafficLight.YELLOW):
            # Predict where will ego be in the future

            secs = 2*TIME_COMPENSATE_PROP_DELAYS #seconds
            
            wp_in_secs, dist_covered = self.get_ego_wp_at_t(secs,
                                                            self.current_velocity, 0)

            # Would we have crossed the light?
            #tl_wp = tl_stopline_wp + (STOP_LINE_TO_TL_DIST/2.)

            dist_to_stop = dist_to_stopline + STOP_LINE_TO_TL_DIST/2. - (CAR_LENGTH/2.)
            
            if(dist_covered < dist_to_stop):
                cur_state = TrafficLight.RED
                rospy.loginfo("      Cannot cross tl xsection. Dist to stop %0.2f. Dist covered %0.2f. Prepare to stop...",
                              dist_to_stop, dist_covered)
            else:
                rospy.loginfo("      Can cross tl xsection. Dist to stop %0.2f. Dist covered %0.2f. Continue on...",
                              dist_to_stop, dist_covered)
                
                accelerate = True

        elif(self.tl_cur_state == TrafficLight.RED
             or self.tl_cur_state == TrafficLight.UNKNOWN):
            # Predict where will ego be in the future
            
            secs = TIME_COMPENSATE_PROP_DELAYS #seconds
            
            wp_in_secs, dist_covered = self.get_ego_wp_at_t(secs,
                                                            self.current_velocity, 0)

            # Would we have crossed the light?
            #tl_wp = tl_stopline_wp
            
            dist_to_stop = dist_to_stopline - (CAR_LENGTH/2.)
            
            # If it just turned Red and we are in the middle of the intersection push forward
            if((dist_covered > dist_to_stop) and (tl_on_for_secs < 2)):
                cur_state = TrafficLight.GREEN
                rospy.loginfo("      Will cross tl xsection. Dist to stop %0.2f. Dist covered %0.2f. Continue on...",
                              dist_to_stop, dist_covered)

            else:
                rospy.loginfo("      Cannot cross tl xsection. Dist to stop %0.2f. Dist covered %0.2f. Prepare to stop...",
                              dist_to_stop, dist_covered)
                
        return cur_state, accelerate, dist_to_stop

    def get_final_wp_vel_accl(self, ego_wp):
        
        dist_to_go = MAX_LOOKAHEAD_DIST
        
        end_vel = MAX_SPEED
        accl = MAX_ACC

        # Dont decelerate if there is no need to...
        if((self.tl_cur_wp_ix - ego_wp) > (MAX_BRAKE_DIST) + 30): # TODO: Macro??
            return dist_to_go, end_vel, accl

        # There is a traffic light ahead. Slow down. (what we do as human drivers...)
        end_vel = (0.8*MAX_SPEED) 
        accl = (0.8*MAX_ACC)
        
        cur_state, accelerate, dist_to_stop = self.get_light_state_to_use(ego_wp, self.tl_cur_wp_ix)
        
        if(cur_state == TrafficLight.RED):
            # Actual Red or treat as Red
            
            if(dist_to_stop < 0):

                dist_to_go = 0
                end_vel = 0
                accl = 0
            else:

                dist_to_go = dist_to_stop
                accl = ((dist_to_stop/MAX_LOOKAHEAD_WPS)*MAX_ACC)
                end_vel = math.sqrt(2*accl*dist_to_stop)
            
            if(end_vel < 0.5):
                end_vel = 0
                                
            if(end_vel > MAX_SPEED):
                end_vel = MAX_SPEED
                
            rospy.loginfo("      RED: dist_to_go=%0.2f, end_vel=%0.2f, accl=%0.2f",
                          dist_to_go, end_vel, accl)

                
        elif(cur_state == TrafficLight.GREEN):
            # Actual Green or treat as Green
            if(accelerate == True):
                end_vel = MAX_SPEED
                accl = MAX_ACC
                
            rospy.loginfo("      GREEN: dist_to_go=%0.2f, end_vel=%0.2f, accl=%0.2f",
                          dist_to_go, end_vel, accl)

        elif(cur_state == TrafficLight.YELLOW):
            # Actual Yellow
            if(accelerate == True):
                end_vel = MAX_SPEED
                accl = MAX_ACC
                
            rospy.loginfo("      YELLOW: dist_to_go=%0.2f, end_vel=%0.2f, accl=%0.2f",
                          dist_to_go, end_vel, accl)


        return dist_to_go, end_vel, accl

    def create_final_waypoints(self, ego_pred_wp, dist_to_go, end_vel, accl):
        #find next waypoint after ego
        # Continue till you have covered the distance

        if(dist_to_go == 0):
            # Brake rightaway
            next_wp = self.find_very_next_wp_after(ego_pred_wp)
            wp = self.base_waypoints.waypoints[next_wp]
            # update longitudinal velocity in waypoint
            wp.twist.twist.linear.x = 0
            
            # append waypoint
            self.final_waypoints.waypoints.append(wp)

            return 1
        
        num_waypoints = 0
        
        next_wp = ego_pred_wp #Initialize

        dist_covered = 0

        v = end_vel
        v2 = end_vel*end_vel

        while(True):
            final_vel_reached = False
            cur_wp = next_wp
            next_wp = self.find_very_next_wp_after(cur_wp)

            step_dist = self.dist_between_waypoints(cur_wp, next_wp)
            
            dist_covered += step_dist

            if(dist_covered >= dist_to_go or step_dist < 0):
                #rospy.loginfo("   *** BREAK. Distance covered %0.4f, step_dist %0.4f",
                #              dist_covered, step_dist)
                break
            else:
                wp = self.base_waypoints.waypoints[next_wp]
                                
                if not final_vel_reached:
                    
                    # propagate velocity from previous waypoint
                    v2 += 2.*accl*step_dist
                    if (v2 > 0.):
                        v = math.sqrt( v2 )
                    else:
                        v = 0.

                    # check whether target velocity has been reached
                    if (accl>0. and v>=end_vel) or (accl<0. and v<=end_vel):
                        v = end_vel
                        final_vel_reached = True

                # update longitudinal velocity in waypoint
                wp.twist.twist.linear.x = v

                # append waypoint
                self.final_waypoints.waypoints.append(wp)

                num_waypoints += 1
                
        #rospy.loginfo("Num waypoints published %d", num_waypoints)
        
        return num_waypoints

        
    # Calculate and publish final waypoints
    def loop(self):

        rate = rospy.Rate(self.freq)

        while not rospy.is_shutdown():

            if self.initialized():

                # initialise final waypoints (guidance)
                self.final_waypoints = Lane()

                # Get the predicted waypoint for ego assuming curret_velocity
                ego_pred_wp, dist_covered = self.get_ego_wp_at_t(TIME_COMPENSATE_PROP_DELAYS,
                                                                 self.current_velocity, 0)

                # Get the ending wp, velocity and accl to publish
                dist_to_go, end_vel, accl = self.get_final_wp_vel_accl(ego_pred_wp)

                num_waypoints = self.create_final_waypoints(ego_pred_wp, dist_to_go, end_vel, accl)

                if(num_waypoints > 0):
                    # publish to topic final_waypoints
                    self.final_waypoints_pub.publish(self.final_waypoints)
                    

            # sleep until next control cycle
            rate.sleep()

    # Distance between two waypoints along waypoint path
    # wp1 and wp2 are waypoint indices
    def dist_ix(self, wp1, wp2):
        d = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            d += dl(
                self.base_waypoints.waypoints[wp1].pose.pose.position,
                self.base_waypoints.waypoints[i].pose.pose.position)
            wp1 = i
        return d


    # Distance between two 'position' elements
    def dist(self, p1, p2):
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return math.sqrt(x*x + y*y + z*z)


#    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
#        pass


    # Print auxiliary functions

    def print_msgHdr(self, header):
        rospy.loginfo("      header:seq:%d time:%d.%d id:%s",
                      header.seq,
                      header.stamp.secs, header.stamp.nsecs,
                      header.frame_id);
        

    def print_poseStamped(self, msg):

        rospy.loginfo("  Pose msg:")
        
        self.print_msgHdr(msg.header)
        
        rospy.loginfo("      position: x:%0.2f y:%0.2f z:%0.2f",
                      msg.pose.position.x,
                      msg.pose.position.y,
                      msg.pose.position.z);
        rospy.loginfo("      orientation: x:%0.2f y:%0.2f z:%0.2f w:%0.2f",
                      msg.pose.orientation.x,
                      msg.pose.orientation.y,
                      msg.pose.orientation.z,
                      msg.pose.orientation.w);


    def print_twist(self, msg):

        rospy.loginfo("  Twist msg:")
        
        self.print_msgHdr(msg.header)
        
        rospy.loginfo("      linear: x:%0.2f y:%0.2f z:%0.2f",
                      msg.twist.linear.x,
                      msg.twist.linear.y,
                      msg.twist.linear.z);
        rospy.loginfo("      angular: x:%0.2f y:%0.2f z:%0.2f",
                      msg.twist.angular.x,
                      msg.twist.angular.y,
                      msg.twist.angular.z);
        

    def print_allWaypoints(self, waypoints):
        
        for ix, ww in enumerate(waypoints):
            rospy.loginfo("Waypoint %d:", ix);
            
            self.print_poseStamped(ww.pose)
            
            self.print_twist(ww.twist)


    def print_allWaypoints_s(self, waypoints):
            
        for ix, ww in enumerate(waypoints):
            rospy.loginfo("Waypoint %d: segment:%0.2f", ix, self.base_waypoints_s[ix]);
            
            self.print_poseStamped(ww.pose)
            
            self.print_twist(ww.twist)


    def print_dbg_light(self, light):
        rospy.loginfo("   Light state=%d", light.state)
        rospy.loginfo("   (unknown=%d, green=%d, yellow=%d, red=%d)",
                      light.UNKNOWN, light.GREEN, light.YELLOW, light.RED)
        self.print_msgHdr(light.header)
        self.print_poseStamped(light.pose)


    def print_dbg_all_lights(self, lights):
        i = 0
        for light in lights:
            rospy.loginfo(" Light:%d", i);
            self.print_dbg_light(light)
            i += 1


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
