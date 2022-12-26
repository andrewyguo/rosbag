import argparse,os,sys,glob
from cv_bridge import CvBridge
import rosbag
import numpy as np
from bisect import bisect_left
import os
import cv2
import tf2_ros
import rospy
import tf
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2
from transformations import *
import copy
from PIL import Image
from tqdm import tqdm 


def extractCameraInfo(msg,):
  W = msg.width
  H = msg.height
  K = np.array(msg.K).reshape(3,3)
  D = np.array(msg.D)
  P = np.array(msg.P).reshape(3,4)
  return H,W,K,D,P


def undistort_rgbd(color,depth,K,D,newH,newW):
  newK,_ = cv2.getOptimalNewCameraMatrix(K, D, (W,H), 0, centerPrincipalPoint=True)
  map1,map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), newK, (W,H), cv2.CV_32FC1)
  color_undist = cv2.remap(color, map1, map2, cv2.INTER_NEAREST)
  depth_undist = cv2.remap(depth, map1, map2, cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)
  return color_undist,depth_undist


def toOpen3dCloud(points,colors=None,normals=None):
  import open3d as o3d
  cloud = o3d.geometry.PointCloud()
  cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
  if colors is not None:
    if colors.max()>1:
      colors = colors/255.0
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
  if normals is not None:
    cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
  return cloud



def depth2xyzmap(depth, K):
	invalid_mask = (depth<0.1)
	H,W = depth.shape[:2]
	vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
	vs = vs.reshape(-1)
	us = us.reshape(-1)
	zs = depth.reshape(-1)
	xs = (us-K[0,2])*zs/K[0,0]
	ys = (vs-K[1,2])*zs/K[1,1]
	pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
	xyz_map = pts.reshape(H,W,3).astype(np.float32)
	xyz_map[invalid_mask] = 0
	return xyz_map.astype(np.float32)


def parsePointCloudMsg(msg):
  rgb = []
  points = []
  for point in sensor_msgs.point_cloud2.read_points(msg, skip_nans=True):
    x = point[0]
    y = point[1]
    z = point[2]
    points.append([x,y,z])
    rgb.append(point[3])
  points = np.array(points)
  rgb = np.array(rgb).astype(np.float32)

  rgb_arr = rgb.copy()
  rgb_arr.dtype = np.uint32
  r = np.asarray((rgb_arr >> 16) & 255, dtype=np.uint8).reshape(-1,1)
  g = np.asarray((rgb_arr >> 8) & 255, dtype=np.uint8).reshape(-1,1)
  b = np.asarray(rgb_arr & 255, dtype=np.uint8).reshape(-1,1)
  colors = np.hstack((r,g,b))
  return points, colors


if __name__ == "__main__":
  is_kinect = True
  kinect_remapped = True
  start_frame = ''
  target_frame = ''
  rgb_topic = '/rgb/image_raw'
  depth_topic = '/depth_to_rgb/image'
  tf_topic = '/tf'
  file_name = '2022-12-21-15-35-38.bag'
  bag_path = '/home/andrewg/rosbags/' +  file_name

  out_dir = bag_path.replace('.bag','')
  rgb_path = os.path.join(out_dir,'rgb')
  depth_path = os.path.join(out_dir,'depth')
  tf_path = os.path.join(out_dir,'tf')
  ply_path = os.path.join(out_dir,'ply')
  icp_path = os.path.join(out_dir,'annotated_poses')
  os.system('mkdir -p {} {} {} {} {}'.format(rgb_path,depth_path,tf_path,ply_path,icp_path))

  bag = rosbag.Bag(bag_path)
  keep_intervals = [(-np.inf, np.inf)]

  iterator = bag.read_messages()

  tf_table = {}
  tf_table_times = {}

  depth_times = []
  rgb_times = []
  frames = set()

  ############## Get camera info
  for topic, msg, t in iterator:
    if '/rgb/camera_info' in topic:
      H,W,K,D,P = extractCameraInfo(msg)
      if (np.array(D)!=0).any():
        map1,map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), K, (W,H), cv2.CV_32FC1)
      else:
        map1,map2 = None,None
      print("K")
      print(K)
      np.savetxt("{}/cam_K.txt".format(out_dir),K)
      break

  iterator = bag.read_messages()
  for topic, msg, t in tqdm(iterator, desc="Processing images from ROS Bag..."):
    t = int(str(t))
    if topic==rgb_topic:
      bridge = CvBridge()
      cv_image = bridge.imgmsg_to_cv2(msg)
      rgb_times.append(t)
      if map1 is not None:
        color = cv2.remap(cv_image, map1, map2, cv2.INTER_LINEAR)[...,:3]
      else:
        color = cv_image.copy()
      color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
      Image.fromarray(color).save(os.path.join(rgb_path,str(t)+'.png'))

    elif topic==depth_topic:
      depth_times.append(t)

    # elif topic=='/tf':
    #   transforms = msg.transforms
    #   for trans in transforms:
    #     frame_id = trans.header.frame_id
    #     if frame_id[0]=='/':
    #       frame_id = frame_id[1:]
    #     child_frame_id = trans.child_frame_id
    #     if child_frame_id[0]=='/':
    #       child_frame_id = child_frame_id[1:]
    #     pose = np.eye(4)
    #     pose[:3,3] = np.array([trans.transform.translation.x,trans.transform.translation.y,trans.transform.translation.z])
    #     q_wxyz = np.array([trans.transform.rotation.w,trans.transform.rotation.x,trans.transform.rotation.y,trans.transform.rotation.z])
    #     pose[:3,:3] = T.quaternion_matrix(q_wxyz)[:3,:3]
    #     if (child_frame_id,frame_id) not in tf_table:
    #         tf_table[(child_frame_id,frame_id)] = {}
    #         tf_table_times[(child_frame_id,frame_id)] = []
    #     tf_table[(child_frame_id,frame_id)][t] = pose.copy()
    #     tf_table_times[(child_frame_id,frame_id)].append(t)


  # if start_frame!='':
  #   for (child_frame_id,frame_id) in tf_table.keys():
  #     frames.add(child_frame_id)
  #     frames.add(frame_id)
  #   print('frames:',frames)
  #   assert start_frame in frames and target_frame in frames

  ###########!DEBUG tmp
  rgb_times = []
  tmp = sorted(os.listdir('{}/rgb/'.format(out_dir)))
  for tt in tmp:
    rgb_times.append(int(tt.replace('.png','')))


  rgb_times = np.array(rgb_times).astype(int)
  depth_times = np.array(depth_times).astype(int)

  for k in tf_table_times.keys():
    tf_table_times[k] = np.array(tf_table_times[k])

  depth_times_sync = []

  ########### Find sync times for other topics
  for t in rgb_times:
    idx = np.argmin(np.abs(depth_times-t))
    best_t = depth_times[idx]
    depth_times_sync.append(best_t)
  depth_times_sync = np.array(depth_times_sync).astype(int)
  print("depth_times_sync#",len(depth_times_sync))

  ########## Dump other topics based on sync times
  iterator = bag.read_messages()
  for topic, msg, t in iterator:
    t = int(str(t))
    print("Dumping {}".format(t))
    if topic==depth_topic:
      if t in depth_times_sync:
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg).copy()
        invalid_mask = np.isnan(cv_image)
        cv_image[invalid_mask] = 0
        if msg.encoding=='32FC1':
          depth = cv_image
        elif msg.encoding=='16UC1':
          depth = cv_image.astype(float)*0.001
        else:
          raise NameError
        depth[depth<0.1] = 0
        depth[depth>2] = 0
        if map1 is not None:
          depth = cv2.remap(depth, map1, map2, cv2.INTER_NEAREST)

        matched_pos = np.where(depth_times_sync==t)[0]  #!NOTE: one can match multiple rgb
        print('matched_pos',type(matched_pos),matched_pos)
        matched_rgb_times = rgb_times[matched_pos]
        for rgb_time in matched_rgb_times:
          print('rgb_time {}, depth_time {}'.format(rgb_time,t))
          cv2.imwrite("{}/{}.png".format(depth_path,rgb_time),(depth*1000).astype(np.uint16))

    # elif topic==tf_topic:
    #   if start_frame!='':
    #     tf = get_tf_branch(tf_table,tf_table_times,start_frame,target_frame,t)
    #     np.savetxt('{}/{}_{}_{}.txt'.format(tf_path,t,start_frame.replace('/','_'),target_frame.replace('/','_')), tf)

  n_color_files = len(glob.glob('{}/rgb/*.png'.format(out_dir)))
  n_depth_files = len(glob.glob('{}/depth/*.png'.format(out_dir)))
  if n_color_files!=n_depth_files:
    raise RuntimeError("{}!={}".format(n_color_files,n_depth_files))

  ######## Make example ply
  for t in rgb_times:
    rgb_file = os.path.join(rgb_path,str(t)+'.png')
    rgb = np.array(Image.open(rgb_file))
    depth = cv2.imread(rgb_file.replace('rgb','depth'),-1)/1e3
    xyz_map = depth2xyzmap(depth,K)
    valid_mask = xyz_map[...,2]>=0.1
    pts = xyz_map[valid_mask].reshape(-1,3)
    colors = rgb[valid_mask].reshape(-1,3)
    pcd = toOpen3dCloud(pts,colors)
    o3d.io.write_point_cloud('{}/ply/{}.ply'.format(out_dir,t),pcd)
    break

  bag.close()