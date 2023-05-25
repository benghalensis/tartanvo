#!/usr/bin/env python

# Software License Agreement (BSD License)
#
# Copyright (c) 2020, Wenshan Wang, Yaoyu Hu,  CMU
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of CMU nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image, CameraInfo, Imu
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from cv_bridge import CvBridge

from Datasets.utils import ToTensor, Compose, CropCenter, DownscaleFlow, make_intrinsics_layer, get_camera_matrix, get_distortion_coef, invert_pose
from Datasets.transformation import se2SE, SO2quat
from TartanVO import TartanVO
import time

import threading
from collections import deque
import pdb

class TartanVONode(object):
    def __init__(self):

        model_name = rospy.get_param('~model_name', 'tartanvo_1914.pkl')
        w = rospy.get_param('~image_width', 640)
        h = rospy.get_param('~image_height', 480)
        fx = rospy.get_param('~focal_x', 320.0)
        fy = rospy.get_param('~focal_y', 320.0)
        ox = rospy.get_param('~center_x', 320.0)
        oy = rospy.get_param('~center_y', 240.0)
        self.cam_intrinsics = [w, h, fx, fy, ox, oy]

        self.cv_bridge = CvBridge()
        self.transform = Compose([CropCenter((448, 640)), DownscaleFlow(), ToTensor()])#, Normalize(mean=[0., 0., 0.],std=[1., 1., 1.])])
        self.intrinsic = make_intrinsics_layer(w, h, fx, fy, ox, oy)
        self.tartanvo = TartanVO(model_name)

        self.pose_pub = rospy.Publisher("tartanvo_pose", PoseStamped, queue_size=10)
        self.odom_pub = rospy.Publisher("tartanvo_odom", Odometry, queue_size=10)
        self.imu_odom_pub = rospy.Publisher("tartanvio_odom", Odometry, queue_size=10)
        rospy.Subscriber('/crl_rzr/multisense_back/aux/image_color', Image, self.handle_img)
        rospy.Subscriber('/crl_rzr/multisense_back/aux/image_color/camera_info', CameraInfo, self.handle_caminfo)
        rospy.Subscriber('/crl_rzr/xsens/imu/data', Imu, self.handle_imu, tcp_nodelay=True)
        rospy.Subscriber('vo_scale', Float32, self.handle_scale)

        self.last_img = None
        self.pose = np.matrix(np.eye(4,4))
        self.imu_pose = np.eye(4)
        self.scale = 1.0

        self.cameraMatrix = None
        self.distCoeffs = None

        self.imu_msg_queue = deque(maxlen=50)
        self.imu_mutex = threading.Lock()
        self.camera_imu_transform = np.eye(4)

    def handle_caminfo(self, msg):
        w = msg.width
        h = msg.height
        fx = msg.K[0]
        fy = msg.K[4]
        ox = msg.K[2]
        oy = msg.K[5]
        new_intrinsics = [w, h, fx, fy, ox, oy]
        change = [xx!=yy for xx,yy in zip(new_intrinsics, self.cam_intrinsics)]

        if self.cameraMatrix is None:
            self.cameraMatrix = get_camera_matrix(msg)
        if self.distCoeffs is None:
            self.distCoeffs = get_distortion_coef(msg)
        if True in change:
            self.intrinsic = make_intrinsics_layer(w, h, fx, fy, ox, oy)
            self.cam_intrinsics = [w, h, fx, fy, ox, oy]
            print('Camera intrinsics updated..')

    def handle_scale(self, msg):
        self.scale = msg.data

    def handle_imu(self, msg):
        # Block that thread
        self.imu_mutex.acquire(blocking=True)
        # Append the IMU message based on max size
        self.imu_msg_queue.append(msg)
        # Release the thread
        self.imu_mutex.release()

    def get_gravity_vec(self, ):
        # Block that thread
        self.imu_mutex.acquire(blocking=True)

        imu_acc = np.zeros(shape=(len(self.imu_msg_queue), 3))
        rospy.loginfo("The length of the imu queue is %d", len(self.imu_msg_queue))
        # Process the messages
        for i in range(imu_acc.shape[0]):
            msg = self.imu_msg_queue.popleft()
            imu_acc[i, 0] = msg.linear_acceleration.x
            imu_acc[i, 1] = msg.linear_acceleration.y
            imu_acc[i, 2] = msg.linear_acceleration.z
        gravity_direction = imu_acc.mean(axis=0)
        rospy.loginfo("The length of the imu queue is %d", len(self.imu_msg_queue))
        # Release the thread
        self.imu_mutex.release()

        return gravity_direction.reshape(-1,1) / np.linalg.norm(gravity_direction)
    
    def align_pose_with_vec(self, pose, vec, axis: int):
        z_axis = pose[0:3,2]
        dot_product = (z_axis.T @ vec).item()
        if dot_product < 0.99:
            return

        # Calculate axis and angle of rotation
        axis = np.cross(z_axis.T, vec.T).flatten()
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(dot_product)

        # Calculate rotation matrix
        skew_symmetric = np.array([[       0, -axis[2],  axis[1]],
                                   [ axis[2],        0, -axis[0]],
                                   [-axis[1],  axis[0],       0]])
        
        R_rot = np.eye(3) + np.sin(angle) * skew_symmetric + (1 - np.cos(angle)) * skew_symmetric @ skew_symmetric

        # Apply rotation
        pose[0:3, 0:3] = R_rot @ pose[0:3, 0:3]

    def handle_img(self, msg):
        starttime = time.time()
        # image_np = self.cv_bridge.imgmsg_to_cv2(msg, "mono8")
        image_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

        # Undistorting the image
        if self.cameraMatrix is None or self.distCoeffs is None:
            print('Waiting for messages from camera_info')
            return
        image_np = cv2.undistort(image_np, self.cameraMatrix, self.distCoeffs)

        if image_np.shape[0] != self.intrinsic.shape[0] or image_np.shape[1] != self.intrinsic.shape[1]:
            print('The intrinsic parameter does not match the image parameter!')
            return

        if self.last_img is not None:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = msg.header.stamp
            pose_msg.header.frame_id = 'map'

            imu_pose_msg = PoseStamped()
            imu_pose_msg.header.stamp = msg.header.stamp
            imu_pose_msg.header.frame_id = 'map'

            sample = {'img1': self.last_img, 
                      'img2': image_np, 
                      'intrinsic': self.intrinsic
                      }
            sample = self.transform(sample)
            sample['img1'] = sample['img1'][None] # increase the dimension
            sample['img2'] = sample['img2'][None]
            sample['intrinsic'] = sample['intrinsic'][None]

            motion, _ = self.tartanvo.test_batch(sample)
            motion = motion[0]
            # adjust the scale if available
            if self.scale!=1:
                trans = motion[:3]
                trans = trans / np.linalg.norm(trans) * self.scale
                motion[:3] = trans
                print(self.scale)

            motion_mat = se2SE(motion)
            self.pose = self.pose @ invert_pose(motion_mat)

            # Convert to IMU frame
            self.imu_pose = self.imu_pose @ invert_pose(motion_mat)

            # Do the gravity alignment
            gravity_direction = self.get_gravity_vec()
            self.align_pose_with_vec(self.imu_pose, gravity_direction, 2)

            quat = SO2quat(self.pose[0:3,0:3])
            imu_quat = SO2quat(self.imu_pose[0:3,0:3])

            pose_msg.pose.position.x = self.pose[0,3]
            pose_msg.pose.position.y = self.pose[1,3]
            pose_msg.pose.position.z = self.pose[2,3]
            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]

            self.pose_pub.publish(pose_msg)     

            odom_msg = Odometry()
            odom_msg.header = pose_msg.header
            odom_msg.pose.pose = pose_msg.pose

            imu_odom_msg = Odometry()
            imu_odom_msg.header = pose_msg.header
            imu_odom_msg.pose.pose.position.x = self.imu_pose[0,3]
            imu_odom_msg.pose.pose.position.y = self.imu_pose[1,3]
            imu_odom_msg.pose.pose.position.z = self.imu_pose[2,3]
            imu_odom_msg.pose.pose.orientation.x = imu_quat[0]
            imu_odom_msg.pose.pose.orientation.y = imu_quat[1]
            imu_odom_msg.pose.pose.orientation.z = imu_quat[2]
            imu_odom_msg.pose.pose.orientation.w = imu_quat[3]

            self.odom_pub.publish(odom_msg)
            self.imu_odom_pub.publish(imu_odom_msg)

        self.last_img = image_np.copy()
        print("    call back time: {}:".format(time.time()-starttime))

if __name__ == '__main__':
    rospy.init_node("tartanvo_node", log_level=rospy.INFO)
    node = TartanVONode()
    rospy.spin()
