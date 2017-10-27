#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32 
from geometry_msgs.msg import PoseStamped, TwistStamped, Point
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray, TrafficLight, TrafficLightStateAndWP

import math
import tf

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
LOOKAHEAD_WPS = 30

LOOKAHEAD_WPS_FOR_TL = 150 

# Max allowed speed (limit set by waypoint_loader)
MAX_SPEED  = rospy.get_param('/waypoint_loader/velocity') *1000/3600  # m/s
rospy.loginfo('MAX_SPEED: %f m/s', MAX_SPEED)

MAX_ACC    = 9.  # m/s^2
BRAKE_DECC = 2.  # nominal decceleration used to stop car [m/s^2]
                 # (ego will be braked at up to MAX_ACC if light suddently turns red)

# Braking distance when ego is at maximum speed and brakes at nominal decceleration
MAX_BRAKE_DIST = MAX_SPEED*MAX_SPEED/(2.*BRAKE_DECC)

MARGIN_TO_TL = 23.               # distance ahead of red light ego tries to stop at [m]
MARGIN_FOR_BRAKE_OVERSHOOT = 7.  # max distance after stop point ego may stop at [m]
                                 # (due to PID characteristic time and latency)  

VEL_THRESHOLD_FOR_YELLOW = 3. #m/s

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

        # index of closest waypoint to red light
        # (set to none if no red light in front of car)
        self.next_red_yellow_tl_wp_ix = None
        
        # current velocity
        self.current_velocity = None

        # last measured stamped pose
        self.meas_pose = None

        # waypoint updater publishing frequency
        self.freq = 1  # Hz

        # flag set to truth first time an image is received
        self.traffic_received = False
        
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


    # Predict ego position at given input time,
    # propagating with last available pose and velocity measurements
    # Last measured heading is also returned
    #
    # If time==Null, last pose is returned, w/o propagation
    def get_pose(self, time):

        # get last measured stamped pose
        pos = self.meas_pose.pose.position

        # calculate ego heading from last measured pose
        #heading = 2.*math.atan2(
        #    self.meas_pose.pose.orientation.z, self.meas_pose.pose.orientation.w)
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

            self.base_waypoints_s.append(0)

            rospy.loginfo('Computing base_waypoint segments...')
            for ix in range(len(waypoints.waypoints)-1):
                self.base_waypoints_s.append(self.dist_ix(ix, ix+1))
        
            for ix in range(len(self.base_waypoints_s)-1):
                self.base_waypoints_s[ix+1] += self.base_waypoints_s[ix]


    # Callback function for /traffic_waypoint message 
    def traffic_cb(self, msg):

        self.next_red_yellow_tl_wp_ix = None

        state = msg.state
        
        if (state == TrafficLight.RED):
            self.next_red_yellow_tl_wp_ix = msg.wp_ix
        elif ((state == TrafficLight.YELLOW)
              and (self.current_velocity >= VEL_THRESHOLD_FOR_YELLOW)):
            self.next_red_yellow_tl_wp_ix = msg.wp_ix
              
        self.traffic_received = True

    def dist_between_waypoints(self, ix1, ix2):
        return (self.base_waypoints_s[ix2] - self.base_waypoints_s[ix1])

    # Append to self.final_waypoints, waypoints to connect an initial state
    # to a final state (including wp for initial state but excluding final state)
    # Velocities are calculated for either constant acceleration or
    # MAX_ACC, depending on value of input flag 'at_max_acc'
    def append_final_waypoints(
        self,
        init_wp_ix, init_velocity,
        final_wp_ix, final_velocity,
        at_max_acc =True ):

        # check input indices
        if final_wp_ix <= init_wp_ix:
            rospy.logwarn(
                "append_final_waypoints: final_wp_ix (%f) <= init_wp_ix (%d)",
                final_wp_ix, init_wp_ix)
            return


        # select acceleration ---

        # case max acceleration
        if at_max_acc:

            if final_velocity > init_velocity:
                acc = MAX_ACC
            else:
                acc = -MAX_ACC

        # case constant acceleration
        else:

            # calculate distance between init and final position,
            # along driving path
            d = self.dist_between_waypoints(init_wp_ix, final_wp_ix)

            if d < 0.1:
                rospy.logwarn(
                    "append_final_waypoints: too small distance between init_wp_ix (%d) and final_wp_ix (%f)",
                    init_wp_ix, final_wp_ix)
                return

            # calculate constant acceleration between initial and final states
            acc = (final_velocity - init_velocity)*(final_velocity + init_velocity)/(2.*d)

            # truncate acceleration to maximum allowed value
            if acc > MAX_ACC:
                acc = MAX_ACC
            elif acc < -MAX_ACC:
                acc = -MAX_ACC


        # initialize velocity
        v  = init_velocity
        v2 = v*v

        final_vel_reached = False

        for i in range( init_wp_ix, final_wp_ix ):

            # get base waypoint
            wp = self.base_waypoints.waypoints[i]

            if not final_vel_reached:

                # calculate distance from previous waypoint
                d = self.dist_between_waypoints(i-1, i)

                # propagate velocity from previous waypoint
                v2 += 2.*acc*d
                if (v2 > 0.):
                    v = math.sqrt( v2 )
                else:
                    v = 0.

                # check whether target velocity has been reached
                if (acc>0. and v>=final_velocity) or (acc<0. and v<=final_velocity):
                    v = final_velocity
                    final_vel_reached = True


            # update longitudinal velocity in waypoint
            wp.twist.twist.linear.x = v

            # append waypoint
            self.final_waypoints.waypoints.append(wp)


    # Append base_waypoints between given indeces to self.final_waypoints,
    # leaving velocity of base_waypoints unchanged
    def append_final_waypoints_base_speed(
        self, init_wp_ix, final_wp_ix ):

        # check input indices
        if final_wp_ix <= init_wp_ix:
            rospy.logwarn(
                "append_final_waypoints_base_speed: final_wp_ix (%f) <= init_wp_ix (%d)",
                final_wp_ix, init_wp_ix)
            return

        for i in range( init_wp_ix, final_wp_ix ):

            # get base waypoint
            wp = self.base_waypoints.waypoints[i]

            # append waypoint
            self.final_waypoints.waypoints.append(wp)


    # calculate index of wp in front of the wp with index target_wp_ix
    # and at a distance >= d
    def find_wp_at_distance_in_front(self, target_wp_ix, d):

        ix = target_wp_ix-1
        while self.dist_between_waypoints(ix, target_wp_ix) < d:
            ix -= 1

        return ix


    # Calculate and publish final waypoints
    def loop(self):

        rate = rospy.Rate(self.freq)

        #rospy.loginfo("braking distance: %f", MAX_BRAKE_DIST)

        while not rospy.is_shutdown():

            if self.initialized():

                # initialise final waypoints (guidance)
                self.final_waypoints = Lane()

                # get current time (class rospy.Time)
                t = rospy.get_rostime()

                # estimate ego position and heading based on last measurement
                ego_position, ego_heading = self.get_pose(t)

                # set init waypoint index to waypoint in front of ego
                ego_next_wp_ix = self.next_waypoint(
                    ego_position, ego_heading)

                # set init wp to wp ahead and init velocity
                # to last measured ego velocity
                init_wp_ix = ego_next_wp_ix
                init_velocity = self.current_velocity

                # flags indicating whether there is a red light ahead
                # and whether ego is in the middle of a crossroads
                # (after stop point but before red light)
                red_light_ahead = False
                in_middle_of_crossroads = False

                # if red light in planning horizon
                if( self.next_red_yellow_tl_wp_ix
                    and 0 < self.next_red_yellow_tl_wp_ix-ego_next_wp_ix < LOOKAHEAD_WPS_FOR_TL ):

                    # set flag to red light ahead
                    red_light_ahead = True

                    # find wp index of start point of crossroads
                    start_crossroads_wp_ix = self.find_wp_at_distance_in_front(
                        self.next_red_yellow_tl_wp_ix, MARGIN_TO_TL)

                    # find wp index of target stop point: MARGIN_TO_TL meters before traffic light
                    stop_wp_ix = self.find_wp_at_distance_in_front(
                        start_crossroads_wp_ix, MARGIN_FOR_BRAKE_OVERSHOOT)

                    curr_brake_dist = self.current_velocity * self.current_velocity/(2.*BRAKE_DECC)
                    
                    # identify wp at which ego shall start braking
                    start_brake_wp_ix = self.find_wp_at_distance_in_front(
                        stop_wp_ix, curr_brake_dist)

                    #rospy.loginfo(
                    #    "Red light! ego_wpix=%d, rl_wpix=%d, cross_rds_wpix=%d, stop_wpix=%d",
                    #    ego_next_wp_ix, self.next_red_yellow_tl_wp_ix,
                    #    start_crossroads_wp_ix, stop_wp_ix)
                    
                    #rospy.loginfo(
                    #    "           start_brk_wpix=%d, curr_vel=%f, brake_dist=%f",
                    #    start_brake_wp_ix, self.current_velocity, curr_brake_dist)
                    

                    # if there is some distance up to the point where braking shall start,
                    # advance car up to that point targeting max speed
                    if ego_next_wp_ix < start_brake_wp_ix:

                        # set target velocity to max velocity
                        # and target waypoint to position at which ego shall start to brake
                        target_wp_ix = (start_brake_wp_ix
                                        if ((start_brake_wp_ix - ego_next_wp_ix)<= LOOKAHEAD_WPS)
                                        else (ego_next_wp_ix + LOOKAHEAD_WPS))
                                        
                        target_velocity = MAX_SPEED
                        
                        #self.final_waypoints = Lane()
                        # calculate and append waypoints from init to target state
                        self.append_final_waypoints(
                            init_wp_ix, init_velocity,
                            target_wp_ix, target_velocity)

                        # update init state (for next call to append_final_waypoints)
                        init_wp_ix = target_wp_ix
                        init_velocity = target_velocity

                        rospy.loginfo(
                            "Red light far ahead, guiding to %f m/s, then to a stop in %d wps",
                            target_velocity, stop_wp_ix-ego_next_wp_ix)


                    # if stop point in front of init ego point,
                    # stop ego at stop point in front of red light
                    if init_wp_ix < stop_wp_ix:
                        
                        # set target velocity to zero
                        # and target waypoint to stop point ahead of traffic light
                        target_wp_ix = (stop_wp_ix
                                        if ((stop_wp_ix - init_wp_ix) <= LOOKAHEAD_WPS)
                                        else (init_wp_ix + LOOKAHEAD_WPS))
                        target_velocity = 0.

                        #self.final_waypoints = Lane()
                        # calculate and append waypoints from init to target state
                        self.append_final_waypoints(
                            init_wp_ix, init_velocity,
                            target_wp_ix, target_velocity,
                            False)  # stop exactly at target_wp_ix, not before

                        if ego_next_wp_ix >= start_brake_wp_ix:
                            rospy.loginfo(
                                "Red light close ahead, guiding to a stop in %d wps",
                                stop_wp_ix-ego_next_wp_ix)

                    # case ego is between start of crossroads and red light,
                    # get out of there!
                    elif init_wp_ix > start_crossroads_wp_ix:

                        # ego is in the midle of crossroads, get out of there!
                        in_middle_of_crossroads = True

 
                    # case ego is after nominal stop point but before start of crossroads:
                    # stay there until green light
                    else:
                        rospy.loginfo(
                            "Red light close ahead at %d wps, ego stopped before crossroads, stay there",
                            self.next_red_yellow_tl_wp_ix-ego_next_wp_ix)


                # if no red light ahead or in the middle of crossroads:
                # progress targetting maximum speed
                if (not red_light_ahead) or in_middle_of_crossroads:

                    # set target velocity to max velocity
                    # and target waypoint LOOKAHEAD_WPS waypoints ahead of ego
                    target_wp_ix = ego_next_wp_ix + LOOKAHEAD_WPS
                    target_velocity = MAX_SPEED

                    #self.final_waypoints = Lane()
                    # calculate and append waypoints from init to target state
                    self.append_final_waypoints(
                        init_wp_ix, init_velocity,
                        target_wp_ix, target_velocity)

                    if not red_light_ahead:
                        rospy.loginfo(
                            "No red light ahead, guiding to %f m/s", target_velocity)
                    elif in_middle_of_crossroads:
                        rospy.loginfo(
                            "Red light close ahead at %d wps, ego in the middle of crossroads, guiding to %f m/s",
                            self.next_red_yellow_tl_wp_ix-ego_next_wp_ix, target_velocity)


                #rospy.loginfo("** Number of wps published = %d",
                #              len(self.final_waypoints.waypoints))

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


    # Get waypoint velocity from message
    # (a scalar, along X in vehcile frame)
    #def get_waypoint_velocity(self, waypoint):
    #    return waypoint.twist.twist.linear.x


    # Set waypoint velocity to message
    # (a scalar, along X in vehcile frame)
    #def set_waypoint_velocity(self, waypoints, waypoint, velocity):
    #    waypoints[waypoint].twist.twist.linear.x = velocity


    # Print auxiliary functions

    def print_msgHdr(self, header):
        rospy.loginfo("      header:seq:%d time:%d.%d id:%s",
                      header.seq,
                      header.stamp.secs, header.stamp.nsecs,
                      header.frame_id);
        

    def print_poseStamped(self, msg):

        rospy.loginfo("  Pose msg:")
        
        self.print_msgHdr(msg.header)
        
        rospy.loginfo("      position: x:%f y:%f z:%f",
                      msg.pose.position.x,
                      msg.pose.position.y,
                      msg.pose.position.z);
        rospy.loginfo("      orientation: x:%f y:%f z:%f w:%f",
                      msg.pose.orientation.x,
                      msg.pose.orientation.y,
                      msg.pose.orientation.z,
                      msg.pose.orientation.w);


    def print_twist(self, msg):

        rospy.loginfo("  Twist msg:")
        
        self.print_msgHdr(msg.header)
        
        rospy.loginfo("      linear: x:%f y:%f z:%f",
                      msg.twist.linear.x,
                      msg.twist.linear.y,
                      msg.twist.linear.z);
        rospy.loginfo("      angular: x:%f y:%f z:%f",
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
            rospy.loginfo("Waypoint %d: segment:%f", ix, self.base_waypoints_s[ix]);
            
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
