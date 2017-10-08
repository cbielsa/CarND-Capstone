#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32 
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math

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
MAX_SPEED = 5  # m/s

class WaypointUpdater(object):

    def __init__(self):

        # Initialise node
        rospy.init_node('waypoint_updater')

        # Construct subscribers
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb)

        # Construct publisher
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.base_waypoints = None

        rospy.spin()


    # Auxiliary functions ============================================

    # Return index of closest waypoint
    # self.base_waypoints shall be defined 
    def closest_waypoint(self, ego_x, ego_y):

        min_dist2 = 1e12
        wp_index = 0  # index of closest waypoing
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


    # Return index of next waypoing
    # self.base_waypoints shall be defined 
    def next_waypoing(self, ego_x, ego_y, ego_heading):

        # find closest waypoint
        closest_wp_index = self.closest_waypoint(ego_x, ego_y)
        closest_wp = self.base_waypoints.waypoints[closest_wp_index]

        # calculate the heading from ego to closest waypoint, in global frame [rad]
        closest_wp_x = closest_wp.pose.pose.position.x
        closest_wp_y = closest_wp.pose.pose.position.y
        closest_wp_heading = math.atan2(closest_wp_y-ego_y, closest_wp_x-ego_x)

        # if closest waypoint is behind ego, take next wp
        angle = abs(ego_heading - closest_wp_heading)
        if angle > math.pi:
            angle = abs(2.*math.pi - angle)

        if angle > math.pi/2.:
            closest_wp_index += 1

        return closest_wp_index


    # Subscriber callback functions ==================================

    # Callback function for /base_waypoints
    def waypoints_cb(self, waypoints):
        
        # TODO: Implement
        if not self.base_waypoints:
            rospy.loginfo('Copying base_waypoints...')
            self.base_waypoints = waypoints


    # Callback function for /current_pose
    def pose_cb(self, poseStamped):

        # if base waypoints are defined
        if self.base_waypoints:

            ego_x = poseStamped.pose.position.x
            ego_y = poseStamped.pose.position.y
            #ego_z = poseStamped.pose.position.z

            # compute ego heading from quaternion
            ego_qz = poseStamped.pose.orientation.z
            ego_qw = poseStamped.pose.orientation.w
            ego_heading = 2.*math.atan2(ego_qz, ego_qw)


            #rospy.loginfo("ego pose x: %f, y: %f, z: %f", ego_x, ego_y, ego_z)
            #rospy.loginfo(
            #    "ego pose qx: %f, qy: %f, qz: %f, qw: %f",
            #    ego_qx, ego_qy, ego_qz, ego_qw)

            #rospy.loginfo(
            #    "ego heading: %f deg",
            #    math.degrees(ego_heading) )

            # calculate index of waypoint in front of ego
            next_wp_index = self.next_waypoing(ego_x, ego_y, ego_heading)

            # fill final_waypoints with first LOOKAHEAD_WPS waypoints ahead of ego
            final_waypoints = Lane()

            for i in range(LOOKAHEAD_WPS):

                # get base waypoint
                wp = self.base_waypoints.waypoints[next_wp_index+i]
                
                # Set wp velocity to max allowed speed [m/s]
                wp.twist.twist.linear.x = MAX_SPEED

                # append modified waypoing to final waypoints
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


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
