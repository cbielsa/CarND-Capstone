#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight, TrafficLightStateAndWP
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import random

import matplotlib.pyplot as plt

STATE_COUNT_THRESHOLD = 3
PROCESS_TL_GROUND_TRUTH = False

WPS_TO_TL_FROM_STOP_LINE = 15 #mts
DISTANCE_FROM_TL_FOR_PREDICTION = 120 #mts

class TLDetector(object):

    def __init__(self):

        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        # Currently used to fake notification of traffic light state using ground truth
        self.lights_ix = []
        self.lights_indices = False
        
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.tl_nearest_wps = [] # list of index of nearest waypoint to traffic lights
        self.wp_to_nearest_stopline_wp = {} # map from waypoint index to nearest tl stopline waypoint index

        self.upcoming_light_pub = rospy.Publisher('/traffic_waypoint',
                                                  TrafficLightStateAndWP,
                                                  queue_size=1)

        self.bridge = CvBridge()

        # construct TL classifier
        self.light_classifier = None
        if not PROCESS_TL_GROUND_TRUTH:
            self.light_classifier = TLClassifier()
        else:
            self.light_classifier = 1  # anything != None

        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()


    def initialized(self):

        if self.waypoints and self.light_classifier and self.pose:
            return True
        else:
            return False


    def pose_cb(self, msg):
        self.pose = msg


    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

        # Processing stop line positions
        # Since this function is called only once, doing this
        # calculation here should be fine
        stop_line_positions = self.config['stop_line_positions']
        rospy.loginfo("stop positions %s %d", stop_line_positions, len(stop_line_positions))
        dl = lambda a, b: math.sqrt((a.x - b[0])**2 + (a.y - b[1])**2) 
        for p in stop_line_positions:
            nearest_wp_idx = 0
            min_dist = 1e12
            for i in range(len(waypoints.waypoints)):
                d = dl(waypoints.waypoints[i].pose.pose.position, p) 
                if d < min_dist:
                    min_dist = d
                    nearest_wp_idx = i
            self.tl_nearest_wps.append(nearest_wp_idx)

        # sort the list in case stop lines were not in order
        self.tl_nearest_wps.sort()

        rospy.loginfo("tl_nearest_wps %s", self.tl_nearest_wps)
        # for p in self.tl_nearest_wps:
        #     rospy.loginfo("tl waypoint %s", waypoints.waypoints[p].pose.pose.position)

        # populate map from waypoint index to nearest stopline waypoint index
        wp_idx = 0
        for stop_wp_idx in self.tl_nearest_wps:
            while wp_idx <= stop_wp_idx:
                self.wp_to_nearest_stopline_wp[wp_idx] = stop_wp_idx
                wp_idx += 1

        while wp_idx < len(waypoints.waypoints):
            self.wp_to_nearest_stopline_wp[wp_idx] = 0 # loop around the track
            wp_idx += 1

        # # Randomly test few waypoints
        # for i in random.sample(range(len(waypoints.waypoints)), 10):
        #     rospy.loginfo("idx:%d stop_idx:%d", i, self.wp_to_nearest_stopline_wp[i])
        # i = self.tl_nearest_wps[0]
        # rospy.loginfo("idx:%d stop_idx:%d", i, self.wp_to_nearest_stopline_wp[i])
    
    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        
        if not self.initialized():
            return

        self.has_image = True
        self.camera_image = msg

        # NOTE: Used to lookup light state from ground truth. Must be updated to use
        # process_traffic_lights function when ready
        if(PROCESS_TL_GROUND_TRUTH):
            light_wp, state = self.process_traffic_lights_ground_truth()
        else:
            light_wp, state = self.process_traffic_lights()
            
        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        notify_state = TrafficLightStateAndWP()
        
        if self.state != state:
            self.state_count = 0
            self.state = state
            self.state_first_detect_time = rospy.get_rostime()
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            #light_wp = (light_wp
            #            if ((state == TrafficLight.RED) or (state == TrafficLight.YELLOW))
            #            else -1)
            self.last_wp = light_wp
            if(light_wp != -1):
                notify_state.header.frame_id = '/tl_waypoint'
                notify_state.header.stamp = rospy.get_rostime()
                notify_state.wp_ix = light_wp
                notify_state.state = state
                notify_state.first_detect_time = self.state_first_detect_time
                
                self.upcoming_light_pub.publish(notify_state)
                
        else:
            if(self.last_wp != -1):
                notify_state.header.frame_id = '/tl_waypoint'
                notify_state.header.stamp = rospy.get_rostime()
                notify_state.wp_ix = self.last_wp
                notify_state.state = self.last_state
                notify_state.first_detect_time = self.state_first_detect_time
                self.upcoming_light_pub.publish(notify_state)
            
        self.state_count += 1


    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        return 0

    def get_light_state(self):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: State of traffic light (red, unknown)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return TrafficLight.UNKNOWN

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8") #bgr8

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def find_closest_tl_in_front(self, ego_waypoint_ix):

        #to handle wraparound
        last_tl_wp = self.tl_nearest_wps[len(self.tl_nearest_wps) - 1]
        
        if(ego_waypoint_ix > last_tl_wp + WPS_TO_TL_FROM_STOP_LINE):
            # We are about to wraparound. return the first wp in list
            closest_light_ix = self.tl_nearest_wps[0]

            # Approximation
            dist_to_tl = (self.dist(self.waypoints.waypoints[closest_light_ix].pose.pose.position,
                                    self.waypoints.waypoints[ego_waypoint_ix].pose.pose.position)
                          + WPS_TO_TL_FROM_STOP_LINE)
        else:
            for light_wp_ix in self.tl_nearest_wps:
                if(ego_waypoint_ix > light_wp_ix + WPS_TO_TL_FROM_STOP_LINE):
                    continue
                closest_light_ix = light_wp_ix
                break

            # Approximation
            dist_to_tl = (closest_light_ix + WPS_TO_TL_FROM_STOP_LINE) - ego_waypoint_ix

        return closest_light_ix, dist_to_tl
        
        
    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        # stop_line_positions = self.config['stop_line_positions']
        # if(self.pose):
        #     car_position = self.get_closest_waypoint(self.pose.pose)

        ego_waypoint_ix = self.find_closest_waypoint_ix(self.pose,
                                                        self.waypoints.waypoints)

        min_dist = 1e12
        closest_light_ix = 0
        closest_light_state = 0

        return_light_ix = -1
        return_light_state = TrafficLight.UNKNOWN

        closest_light_ix, dist_to_tl = self.find_closest_tl_in_front(ego_waypoint_ix)

        #rospy.loginfo(" Ego index: %d pos: x:%f, y:%f",
        #              ego_waypoint_ix,
        #              self.pose.pose.position.x,
        #              self.pose.pose.position.y)

        #rospy.loginfo(" Upcoming Light index: %d",
        #              closest_light_ix)

        return_light_ix = closest_light_ix

        # try to detect only if ego is within 120 mts before light
        if( dist_to_tl < DISTANCE_FROM_TL_FOR_PREDICTION): 
            return_light_state = self.get_light_state()

        return return_light_ix, return_light_state
    #===========================================================================

    # ==========================================================================
    # Auxillary functions to implement process tl from ground_truth
    #
    def traffic_cb(self, msg):

        if not self.initialized():
            return

        self.lights = msg.lights

        # NOTE: Here we try to find the indices of the traffic lights
        #       and save them in lights_ix. Even though this is a function
        #       used to fake trffic_waypoint notification in absence of real
        #       tl_detection, making sure that we dont have to lookup the ix
        #       in every iteration
        #       Also note that self.lights can be used to verify training of
        #       for tl_detection using images.
        if(self.lights_indices == False):

            for ix, light in enumerate(self.lights):

                # Find and save index of every light
                
                #rospy.loginfo(" Light:%d", ix);
                #rospy.loginfo("   Light state=%d", light.state)
                #rospy.loginfo("   (unknown=%d, green=%d, yellow=%d, red=%d)",
                #              light.UNKNOWN, light.GREEN, light.YELLOW, light.RED)
                #rospy.loginfo("   Pose msg:")
                #rospy.loginfo("      position: x:%f y:%f z:%f",
                #              light.pose.pose.position.x,
                #              light.pose.pose.position.y,
                #              light.pose.pose.position.z);
                #rospy.loginfo("      orientation: x:%f y:%f z:%f w:%f",
                #              light.pose.pose.orientation.x,
                #              light.pose.pose.orientation.y,
                #              light.pose.pose.orientation.z,
                #              light.pose.pose.orientation.w);

                light_ix = []
                light_ix.append(light)
                light_wp = self.find_closest_waypoint_ix(light.pose, self.waypoints.waypoints)
                light_ix.append(light_wp)

                self.lights_ix.append(light_ix)
                
                #rospy.loginfo("      Waypoint Index: %d", light_wp)

            self.lights_indices = True
        
        else:

            # Update the state of the light
            for ix, light in enumerate(self.lights):
                self.update_light_state(light, self.lights_ix)

    
    def dist(self, p1, p2):
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return math.sqrt(x*x + y*y + z*z)


    def find_closest_waypoint_ix(self, pose, waypoints):
        closest_dist = 999
        light_wp_index = 0
        
        for index, waypoint in enumerate(waypoints):
            
            distance = self.dist(waypoint.pose.pose.position,
                                 pose.pose.position)
            
            if distance < closest_dist:
                light_wp_index = index
                closest_dist = distance

        return light_wp_index


    def update_light_state(self, new_light, lights_ix):
        for light in lights_ix:
            if((light[0].pose.pose.position.x == new_light.pose.pose.position.x)
               and (light[0].pose.pose.position.y == new_light.pose.pose.position.y)):
               light[0].state = new_light.state
               break
    

    #processes traffic light using ground truth data and light postions from lights msg in traffic_cb
    # note - this is different than process_traffic_lights() where 'stop line positions' are used
    def process_traffic_lights_ground_truth(self):

        #  1. get closest_waypoint_index to ego_vehicle
        #  2. compare with closest waypoint_indeices of save light data
        #     and return the closest

        ego_waypoint_ix = self.find_closest_waypoint_ix(self.pose,
                                                        self.waypoints.waypoints)

        min_dist = 1e12
        closest_light_ix = 0
        closest_light_state = 0

        return_light_ix = -1
        return_light_state = TrafficLight.UNKNOWN
        
        # TODO: Fix wraparound
        for light in self.lights_ix:
            if(ego_waypoint_ix > light[1]):
               continue;
            dist_from_light = light[1] - ego_waypoint_ix
            #rospy.loginfo("  ego_ix=%d, light_ix=%d", ego_waypoint_ix, light[1])
            #rospy.loginfo("     min_dist=%d, dist_form_light=%d",
            #              min_dist, dist_from_light)
            if(min_dist > dist_from_light):
               min_dist = dist_from_light
               closest_light_ix = light[1]
               closest_light_state = light[0].state


        #rospy.loginfo(" Ego index: %d pos: x:%f, y:%f",
        #              ego_waypoint_ix,
        #              self.pose.pose.position.x,
        #              self.pose.pose.position.y)

        #rospy.loginfo(" Upcoming Light index: %d",
        #              closest_light_ix)


        # Since this is just faking the detection/state of the light based on ground_truth
        # and the system keeps track of all lights on the track at any given time,
        # notify only if the lights are within a visible range. approximating.

        if ((closest_light_ix - ego_waypoint_ix) < 150):
            return_light_ix = closest_light_ix
            return_light_state = closest_light_state
        
        return return_light_ix, return_light_state
    #===========================================================================
    
if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
