#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32 
from geometry_msgs.msg import PoseStamped, TwistStamped, Point
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray

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

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_SPEED  = 7.  # m/s
MAX_ACC    = 9.  # m/s^2
BRAKE_DECC = 3.  # decceleration used to stop car whenever possible [m/s^2]

# Braking distance when ego is at maximum speed and brakes at max accel.
MAX_BRAKE_DIST = MAX_SPEED*MAX_SPEED/(2.*BRAKE_DECC)

MARGIN_TO_TL = 25.  # distance ahead of red light ego stops at [m]

class WaypointUpdater(object):

    def __init__(self):

        # Initialise node
        rospy.init_node('waypoint_updater')

        # Construct subscribers
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        #rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb)

        # Construct publisher
        self.final_waypoints_pub = rospy.Publisher(
            '/final_waypoints', Lane, queue_size=1)

        # other member variables
        self.base_waypoints = None
        self.final_waypoints = None

        #self.base_waypoints_s = [] # base_waypoints segment numbers

        # index of closest waypoint to red light
        # (set to none if no red light in front of car)
        self.next_red_tl_wp_ix = None
        
        # current velocity
        self.current_velocity = None

        # last measured stamped pose
        self.meas_pose = None

        # waypoint updater publishing frequency
        self.freq = 2  # Hz
        
        # Start waypoint updater loop
        self.loop()


    # Auxiliary functions ============================================

    # Return True if updater has received all messages needed
    # to calculate final waypoints
    def initialized(self):

        if self.base_waypoints and self.current_velocity and self.meas_pose and ( rospy.get_rostime()!=0 ):
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
        heading = 2.*math.atan2(
            self.meas_pose.pose.orientation.z, self.meas_pose.pose.orientation.w)

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


    def get_euler_from_quaternion(quaternion):

        # compute ego heading from quaternion
        qx = quaternion.x
        qy = quaternion.y
        qz = quaternion.z
        qw = quaternion.w
        
        #ego_heading = 2.*math.atan2(ego_qz, ego_qw)
        euler = tf.transformations.euler_from_quaternion([qx,
                                                          qy,
                                                          qz,
                                                          qw])
        return euler


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


    # Callback function for /traffic_waypoint message 
    def traffic_cb(self, msg):

        # case no red light ahead detected
        if(msg.data == -1):
            self.next_red_tl_wp_ix = None
        
        # case red light detected
        else:
            #rospy.loginfo("traffic_cb: Traffic light RED. waypoint index %d", msg.data)
            self.next_red_tl_wp_ix = msg.data


    # Append to self.final_waypoints, waypoints to connect an initial state
    # to a final state (including wp for initial state but excluding
    # final state)
    # Velocities are calculated for constant acceleration
    def append_final_waypoints(
        self,
        init_wp_ix, init_velocity,
        final_wp_ix, final_velocity,
        at_max_acc =True ):

        # check input indices
        if final_wp_ix <= init_wp_ix:
            rospy.loginfo(
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
            d = self.dist_ix(init_wp_ix, final_wp_ix)

            if d < 0.1:
                rospy.loginfo(
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

        #first_iter = True
        first_iter = False

        final_vel_reached = False

        for i in range( init_wp_ix, final_wp_ix ):

            # get base waypoint
            wp = self.base_waypoints.waypoints[i]

            if first_iter:

                first_iter = False

            elif not final_vel_reached:

                # calculate distance from previous waypoint
                d = self.dist_ix(i-1, i)

                # propagate velocity from previous waypoint
                v2 += 2.*acc*d
                if v2 > 0.:
                    v = math.sqrt( v2 )
                else:
                    v = 0.

                # check whether target velocity has been reached
                if (acc>0. and v>final_velocity) or (acc<0. and v<final_velocity):
                    v = final_velocity
                    final_vel_reached = True


            # update longitudinal velocity in waypoint
            wp.twist.twist.linear.x = v

            # append waypoint
            self.final_waypoints.waypoints.append(wp)


    # calculate index of wp in front of the wp with index target_wp_ix
    # and at a distance >= d
    def find_wp_at_distance_in_front(self, target_wp_ix, d):

        ix = target_wp_ix-1
        while self.dist_ix(ix, target_wp_ix) < d:
            ix -= 1

        return ix


    # Calculate and publish final waypoints
    def loop(self):

        rate = rospy.Rate(self.freq)

        rospy.loginfo("braking distance: %f", MAX_BRAKE_DIST)

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

                # if red light in planning horizon
                if( self.next_red_tl_wp_ix
                    and 0 < self.next_red_tl_wp_ix-ego_next_wp_ix < LOOKAHEAD_WPS ):

                    # set target stop point to MARGIN_TO_TL meters before traffic light
                    stop_wp_ix = self.find_wp_at_distance_in_front(
                        self.next_red_tl_wp_ix, MARGIN_TO_TL)

                    # identify wp at which ego shall start braking
                    start_brake_wp_ix = self.find_wp_at_distance_in_front(
                        stop_wp_ix, MAX_BRAKE_DIST)

                    # if there is some distance up to the point where braking shall start,
                    # advance car up to that point targeting max speed
                    if start_brake_wp_ix > ego_next_wp_ix:

                        # set target velocity to max velocity
                        # and target waypoint to position at which ego shall start to brake
                        target_wp_ix = start_brake_wp_ix
                        target_velocity = MAX_SPEED

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


                    # stop in front of the red light
                    
                    # set target velocity to zero
                    # and target waypoint to stop point ahead of traffic light
                    target_wp_ix = stop_wp_ix
                    target_velocity = 0.

                    if target_wp_ix > init_wp_ix:

                        # calculate and append waypoints from init to target state
                        self.append_final_waypoints(
                            init_wp_ix, init_velocity,
                            target_wp_ix, target_velocity,
                            False)  # stop exactly at target_wp_ix, not before

                        if start_brake_wp_ix <= ego_next_wp_ix:
                            rospy.loginfo(
                                "Red light close ahead, guiding to a stop in %d wps",
                                stop_wp_ix-ego_next_wp_ix)

                    else:
                        rospy.loginfo(
                            "Red light close ahead at %d wps, stop point overshot, just stay there :S",
                            self.next_red_tl_wp_ix-ego_next_wp_ix)



                # if no red light ahead or stop point was overshot:
                # progress targetting maximum speed
                else:

                    # set target velocity to max velocity
                    # and target waypoint LOOKAHEAD_WPS waypoints ahead of ego
                    target_wp_ix = ego_next_wp_ix + LOOKAHEAD_WPS
                    target_velocity = MAX_SPEED

                    # calculate and append waypoints from init to target state
                    self.append_final_waypoints(
                        init_wp_ix, init_velocity,
                        target_wp_ix, target_velocity)

                    rospy.loginfo(
                        "No red light ahead, guiding to %f m/s", target_velocity)


                # publish to topic final_waypoints
                #rospy.loginfo("Publishing %d waypoints...",
                #    len(self.final_waypoints.waypoints) )
                self.final_waypoints_pub.publish(self.final_waypoints)

            # sleep until next control cycle
            rate.sleep()


        
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
