from styx_msgs.msg import TrafficLight

import numpy as np
import os
import sys
import tensorflow as tf
import rospy

import sys
import label_map_util as label_map_util

CURR_PATH = os.path.dirname(os.path.realpath(__file__))
# Model used
MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = CURR_PATH + '/tl_model_sim/output_inference_graph/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = CURR_PATH + '/tl_model_sim/tl_label_map.pbtxt'
NUM_CLASSES = 4

class TLClassifier(object):
    def __init__(self):
        self.detection_graph = tf.Graph()
        # load frozen model to memory
        with self.detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self.sess = tf.Session(graph=self.detection_graph)
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represents the level of confidence for each of the objects.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_np = np.asarray(image, dtype = np.uint8)#
        #print("Numpy image shape and dtype is ", image_np.shape, image_np.dtype)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        #print("det scores ", scores)
        #print("det classes ", classes)
        #print("num detections ", num)
        print_classes = np.squeeze(classes).astype(np.int32)
        lbl = self.category_index[print_classes[0]]['name']
        rospy.loginfo("Tensor flow classified label is %s", lbl)
        #print("classified label is ", lbl)
        tl_state = TrafficLight.UNKNOWN
        if lbl == 'Red':
            tl_state = TrafficLight.RED
        elif lbl == 'Yellow':
            tl_state = TrafficLight.YELLOW
        elif lbl == 'Green':
            tl_state = TrafficLight.GREEN
        else:
            tl_state = TrafficLight.UNKNOWN

        return tl_state

        
