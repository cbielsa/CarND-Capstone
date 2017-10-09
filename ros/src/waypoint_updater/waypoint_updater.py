#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32 
from geometry_msgs.msg import PoseStamped
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
MAX_SPEED = 10  # m/s

class WaypointUpdater(object):

    def __init__(self):

        # Initialise node
        rospy.init_node('waypoint_updater')

        # Construct subscribers
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb)


        # For debugging purposes, subscribe to /vehicle/traffic_lights topic
        #  It provides us with traffic lights ground truth.
        # NOTE: Must be commented out while testing with traffic light detection code
        rospy.Subscriber('vehicle/traffic_lights', TrafficLightArray, self.tl_ground_truth_cb)
        
        
        # Construct publisher
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.base_waypoints = None

        self.tl_ground_truth = None
        rospy.spin()


    # Auxiliary functions ============================================

    # Return index of closest waypoint
    # self.base_waypoints shall be defined 
    def closest_waypoint(self, ego_x, ego_y):

        min_dist2 = 1e12
        wp_index = 0  # index of closest waypoint
        index = 0

        for waypoint in self.base_waypoints.waypoints:

            wp_x = waypoint.pose.pose.position.x
            wp_y = waypoint.pose.pose.position.y

            dist2 = (ego_x-wp_x)**2 + (ego_y-wp_y)**2

            if dist2 < min_dist2:
                wp_index = index
                min_dist2 = dist2

            index += 1

        return wp_index


    def get_euler_from_quaternion(self, quaternion):
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
        # if closest waypoint is behind ego, take next wp
        angle = abs(ego_heading - other_heading)
        if angle > math.pi:
            angle = abs(2.*math.pi - angle)

        if angle > math.pi/2.:
            return True
        
        return False
    
    # Return index of next waypoint
    # self.base_waypoints shall be defined 
    def next_waypoint(self, pose, ego_heading):

        ego_x = pose.position.x
        ego_y = pose.position.y
        
        # find closest waypoint
        closest_wp_index = self.closest_waypoint(ego_x, ego_y)
        closest_wp = self.base_waypoints.waypoints[closest_wp_index]

        # calculate the heading from ego to closest waypoint, in global frame [rad]
        closest_wp_x = closest_wp.pose.pose.position.x
        closest_wp_y = closest_wp.pose.pose.position.y
        closest_wp_heading = math.atan2(closest_wp_y-ego_y, closest_wp_x-ego_x)

        
        # if closest waypoint is behind ego, take next wp
        if (True == self.is_behind_ego(ego_heading, closest_wp_heading)):
            closest_wp_index += 1
            
        return closest_wp_index

    # Returns the next light in front of the ego_vehicle.
    # If no light or the status unknown, returns false
    def dbg_get_next_light(self, lights, ego_pose, ego_heading):

        closest_light = None
        found = False
        closest_to_ego = 999
        
        for light in lights:
            dist_from_ego = self.dist(ego_pose.position, light.pose.pose.position)
            light_euler = self.get_euler_from_quaternion(light.pose.pose.orientation)
            light_orientation = light_euler[2]

            light_behind = self.is_behind_ego(ego_heading, light_orientation)

            if(True == light_behind):
                continue;

            if(dist_from_ego < closest_to_ego):
                dist_from_ego = closest_to_ego
                closest_light = light
                found = True

        if ((closest_light != None) and (closest_light.state == closest_light.UNKNOWN)):
            found = False
        
        return found, closest_light
    
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

    # Subscriber callback functions ==================================

    def tl_ground_truth_cb(self, traffic_light):
        self.tl_ground_truth = traffic_light.lights
        #self.print_dbg_all_lights(self.tl_ground_truth)

        
    # Callback function for /base_waypoints
    def waypoints_cb(self, waypoints):
        if not self.base_waypoints:
            rospy.loginfo('Copying base_waypoints...')
            self.base_waypoints = waypoints


    # Callback function for /current_pose
    def pose_cb(self, poseStamped):

        # if base waypoints are defined
        if self.base_waypoints:

            euler = self.get_euler_from_quaternion(poseStamped.pose.orientation)

            ego_heading = euler[2]
            #rospy.loginfo("ego pose x: %f, y: %f, z: %f", ego_x, ego_y, ego_z)
            #rospy.loginfo(
            #    "ego pose qx: %f, qy: %f, qz: %f, qw: %f",
            #    ego_qx, ego_qy, ego_qz, ego_qw)

            #rospy.loginfo(
            #    "ego heading: %f deg", math.degrees(ego_heading))

            # calculate index of waypoint in front of ego
            next_wp_index = self.next_waypoint(poseStamped.pose, ego_heading)

            if (self.tl_ground_truth):
                (light_ahead, light) = self.dbg_get_next_light(self.tl_ground_truth,
                                                               poseStamped.pose,
                                                               ego_heading)
                
                if(light_ahead == True):
                    rospy.loginfo(" ** Light Ahead **")
                    #self.print_dbg_light(light)
                
            # fill final_waypoints with first LOOKAHEAD_WPS waypoints ahead of ego
            final_waypoints = Lane()

            for i in range(LOOKAHEAD_WPS):

                # get base waypoint
                wp = self.base_waypoints.waypoints[next_wp_index+i]
                
                # Set wp velocity to max allowed speed [m/s]
                wp.twist.twist.linear.x = MAX_SPEED

                # append modified waypoint to final waypoints
                final_waypoints.waypoints.append(wp)

            # publish to topic final_waypoints
            self.final_waypoints_pub.publish(final_waypoints)


    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass


    # Get waypoint velocity from message
    # (a scalar, along X in vehcile frame)
    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x


    # Set waypoint velocity to message
    # (a scalar, along X in vehcile frame)
    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity


    # Distance between two waypoints
    # wp1 and wp2 are waypoint indices
    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    #Note that this is different than above. wp1 and wp2 are waypoints not indices.
    def dist(self, p1, p2):
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return math.sqrt(x*x + y*y + z*z)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
