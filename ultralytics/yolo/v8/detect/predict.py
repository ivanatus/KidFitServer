## Ultralytics YOLO 🚀, GPL-3.0 license

import hydra #project configuration dependency
import torch
import argparse
import time
from pathlib import Path

#computer vision dependencies
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque, defaultdict
import numpy as np

import os
import csv
import math

#provide path to globals.py file
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "deep_sort_pytorch", "deep_sort", "sort"))
from globals import Globals

#Firebase setup - NOT NEEDED
import firebase_admin
from firebase_admin import credentials
json_path = os.path.join(BASE_DIR, "newServiceAccountKey.json")
cred = credentials.Certificate(json_path)
#initialization of relevant Firebase services (storage and database)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://exercise-project-bc0dc-default-rtdb.europe-west1.firebasedatabase.app/',
    'storageBucket': 'exercise-project-bc0dc.appspot.com'
})
from firebase_admin import storage
from firebase_admin import db

#dependencies for import of output values and their visualization
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------------------
FEATURES = 2000            # ORB features for frame matching
GOOD_MATCH_RATIO = 0.75    # Lowe ratio for descriptor matching
HISTORY_WINDOW = 30        # frames window to estimate floor baseline
FLOOR_PERCENTILE = 90      # percentile of bottom y-values considered the floor baseline
MIN_POINTS_FOR_FLOOR = 5   # require at least this many bottom points across window
SMOOTH_VEL_WINDOW = 3      # smooth velocities across this many samples
SCALE_X = 0.005            # lateral scaling factor for pseudo-world x (tweak to change unit scale)
# -----------------------------------------------------------------------------------------------
# global containers
_homography_ref = None     # cumulative homography mapping frames to reference
_prev_gray = None         # previous grayscale frame
_orb = cv2.ORB_create(FEATURES)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
# store per-object trajectories in *reference frame* pixel coords
traj_ref = defaultdict(lambda: deque(maxlen=64))       # id -> deque of (x_ref, y_ref, frame_idx, timestamp)
traj_world = defaultdict(lambda: deque(maxlen=64))     # id -> deque of (x_world, y_world, depth, frame_idx, timestamp)
# sliding buffer of recent bottom y-values (in reference frame)
recent_bottoms_ref = deque(maxlen=HISTORY_WINDOW)
# Global storage for previous positions
object_states = {}  # {id: {'frame': int, 'x': float, 'y': float, 'depth': float}}



palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {} #dictionary to store data for tracking

deepsort = None #variable to store the DeepSort tracker instance
frame_class = -1

global_instance = Globals()

def init_tracker(): 
    """Initialize the DeepSort tracker with configuration settings.
    Create csv file for record of movement output values.
    """ 
    global deepsort
    cfg_deep = get_config()
    config_path = os.path.join(BASE_DIR, "deep_sort_pytorch", "configs", "deep_sort.yaml")
    cfg_deep.merge_from_file(config_path)
    checkpoint_path = os.path.join(BASE_DIR, "deep_sort_pytorch", "deep_sort", "deep", "checkpoint", "ckpt.t7")
    print("Checkpoint file exists ", os.path.exists(checkpoint_path), ", checkpoint path: ", checkpoint_path)

    deepsort= DeepSort(checkpoint_path,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
##########################################################################################

def estimate_homography(prev_gray, cur_gray):
    if prev_gray is None or cur_gray is None:
        return None
    kp1, des1 = _orb.detectAndCompute(prev_gray, None)
    kp2, des2 = _orb.detectAndCompute(cur_gray, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None
    matches = _bf.knnMatch(des1, des2, k=2)
    good = []
    for m_n in matches:
        if len(m_n) < 2: 
            continue
        m, n = m_n
        if m.distance < GOOD_MATCH_RATIO * n.distance:
            good.append(m)
    if len(good) < 8:
        return None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)  # maps cur -> prev
    return H


def update_cumulative_homography(H_cur_to_prev):
    global _homography_ref
    if H_cur_to_prev is None:
        return
    if _homography_ref is None:
        # if no cumulative yet, set to prev identity * H_cur_to_prev
        _homography_ref = H_cur_to_prev.copy()
    else:
        # new cumulative = previous_cumulative ∘ H_cur_to_prev
        # because H_cur_to_ref = H_prev_to_ref * H_cur_to_prev
        _homography_ref = _homography_ref @ H_cur_to_prev


def map_point_to_ref(u, v):
    global _homography_ref
    if _homography_ref is None:
        return float(u), float(v)
    pt = np.array([[float(u), float(v)]], dtype=np.float32)
    dst = cv2.perspectiveTransform(pt.reshape(1,1,2), _homography_ref).reshape(2)
    return float(dst[0]), float(dst[1])


def estimate_floor_baseline():
    # recent_bottoms_ref contains y-values in reference frame (pixel coords)
    if len(recent_bottoms_ref) < MIN_POINTS_FOR_FLOOR:
        return None
    arr = np.array(recent_bottoms_ref)
    # use a high percentile (closer to image bottom) as floor baseline
    y_floor = np.percentile(arr, FLOOR_PERCENTILE)
    # also estimate y_top (near horizon / far limit) using low percentile
    y_top = np.percentile(arr, 5) if len(arr) >= MIN_POINTS_FOR_FLOOR else arr.min()
    if math.isclose(y_floor, y_top):
        return None
    return float(y_floor), float(y_top)


def refpixel_to_world(u_ref, v_ref, y_floor, y_top, img_center_x):
    # normalize depth: 0 -> nearest (y == y_floor), 1 -> farthest (y == y_top)
    # clamp to [0,1]
    depth_norm = (y_floor - v_ref) / (y_floor - y_top)
    depth_norm = max(0.0, min(1.0, depth_norm))
    # pseudo-world coordinates:
    x_world = (u_ref - img_center_x) * (depth_norm * SCALE_X)
    y_world = depth_norm  # longitudinal axis = depth_norm (0..1)
    return x_world, y_world, depth_norm


def smooth_velocity(id, vx, vy):
    dq = traj_world[id]
    if len(dq) < SMOOTH_VEL_WINDOW + 1:
        return vx, vy
    # compute last N velocities and average them
    vlist = []
    # reconstruct velocities from consecutive samples
    samples = list(dq)[0:SMOOTH_VEL_WINDOW+1]  # newest first
    for i in range(len(samples)-1):
        p0 = samples[i+1]
        p1 = samples[i]
        dt = (p1[3] - p0[3]) if len(p1) > 3 and len(p0) > 3 else 1/30.0
        if dt == 0:
            continue
        vx_i = (p1[0] - p0[0]) / dt
        vy_i = (p1[1] - p0[1]) / dt
        vlist.append((vx_i, vy_i))
    if not vlist:
        return vx, vy
    vx_avg = sum(v[0] for v in vlist) / len(vlist)
    vy_avg = sum(v[1] for v in vlist) / len(vlist)
    return vx_avg, vy_avg
##########################################################################################

def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values.
    Converts from xyxy format to xywh format (coordinates x and y, width and height).
    """
    #box coordinates
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    #center coordinates
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    """Calculates tlhw format from xyxy bounding box coordinates.
    Tlhw format stands for top left corner coordinates with width and height of box.
    """
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
        
    return tlwh_bboxs

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    """Method for drawing border with rounded corners and rectangles."""
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)

    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    """Method for drawing a UI box on the image"""
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_boxes(img, bbox, names,object_id, identities=None, offset=(0, 0)):
    """Method for drawing boxes, labels and trails on frame image based
    on objects detected in that frame.
    """    
    #cv2.line(img, line[0], line[1], (46,162,112), 3)

    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:  
          data_deque[id] = deque(maxlen= 64)
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)
        
        if(obj_name != 'person'):
            return None
            
        # add center to buffer
        data_deque[id].appendleft(center)
        UI_box(box, img, label=label, color=color, line_thickness=2)
        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
    return img

class DetectionPredictor(BasePredictor):
    """Class for making predictions using YOLO model and post-processing
    of results with DeepSort.
    Includes pre-processing, post-processing and writing out results.
    """
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)
        
        for i, pred in enumerate(preds):
            for det in pred:
                if not int(det[5]) == 0 and not int(det[5]) == 17:
                    if preds.__contains__(pred):
                        preds.remove(pred)
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds


    
    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        all_outputs = []
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
            global_instance.set_global_frame(frame)
        else:
            frame = getattr(self.dataset, 'frame', 0)
            global_instance.set_global_frame(frame)
        
        self.data_path = p
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        if idx >= len(preds):
            global_instance.set_no_of_people(0)
            print("Nista nije detektirano u frame-u " + str(global_instance.get_global_frame()))
            return log_string
        
        det = preds[idx]
        all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            global_instance.set_no_of_people(f"{n}")
            check_string = f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}"
            string_parts = check_string.split(' ')
            if string_parts[1] == 'person' or string_parts[1] == 'persons':
                log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        for *xyxy, conf, cls in reversed(det):
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))
        
            
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)
          
        outputs = deepsort.update(xywhs, confss, oids, im0)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            
            #draw_boxes(im0, bbox_xyxy, self.model.names, object_id,identities)
            # =========================
            # Dynamic floor estimation
            # =========================
            # Approximate floor line using the lowest bbox bottom in this frame
            bottom_ys = [int(bbox[3]) for bbox in bbox_xyxy]
            if len(bottom_ys) > 0:
                y_floor = max(bottom_ys)
            else:
                y_floor = im0.shape[0]  # fallback: bottom of image

            # Precompute horizon line (can be dynamically adapted or kept fixed)
            y_horizon = int(0.2 * im0.shape[0])  # 20% from top — heuristic

            # Draw bounding boxes as before
            draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities)

            # =========================
            # Compute movement per object
            # =========================
            rows_to_write = []

            for i, bbox in enumerate(bbox_xyxy):
                obj_id = int(object_id[i])
                identity = int(identities[i])
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2  # center y (can also use y2 for floor contact)
                bottom_y = y2

                # Relative depth based on pixel row
                depth = 1.0 / max(1, (y_floor - bottom_y + 1))  # inverse distance approx
                depth_norm = depth / (1.0 / max(1, (y_floor - y_horizon)))  # normalize

                # Estimate velocities
                prev_state = object_states.get(obj_id)
                vx = vy = speed = 0.0
                if prev_state is not None:
                    dt = max(1, frame - prev_state['frame'])  # frame gap
                    vx = (cx - prev_state['x']) / dt
                    vy = (cy - prev_state['y']) / dt
                    speed = math.sqrt(vx**2 + vy**2)

                # Update object state
                object_states[obj_id] = {'frame': frame, 'x': cx, 'y': cy, 'depth': depth_norm}

                # Add to write list
                rows_to_write.append({
                    'frame': frame,
                    'object_id': obj_id,
                    'x': round(cx, 2),
                    'y': round(cy, 2),
                    'depth': round(depth_norm, 4),
                    'velocity_x': round(vx, 4),
                    'velocity_y': round(vy, 4),
                    'speed': round(speed, 4)
                })

            # =========================
            # Write to CSV
            # =========================
            csv_name = global_instance.current_video_file + '_movement.csv'
            with open(csv_name, 'a', newline='') as csvfile:
                fieldnames = ['frame', 'object_id', 'x', 'y', 'depth', 'velocity_x', 'velocity_y', 'speed']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for row in rows_to_write:
                    writer.writerow(row)
            
        return log_string
    

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    print("Enters main function")
    """Main method for making predictions using the DetectionPredictor class"""
    #initialize DeepSort tracker and retrieve video from Firebase Storage
    init_tracker()
    #define parameters for YOLO model as well as the source video
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    #source_directory = hydra.utils.to_absolute_path(cfg.source)
    source_directory = os.path.join(BASE_DIR, cfg.source)
    for filename in os.scandir(source_directory):
        print("Filename ", filename)
        if filename.is_file() and filename.name.endswith('.mp4'):
            cap = cv2.VideoCapture(filename.path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration_sec = total_frames / fps 
            print(f"Total frames: {total_frames}, FPS: {fps}, Duration: {duration_sec:.2f}s")
            
            predictor = DetectionPredictor(cfg)
            predictor(filename.path)
            global_instance.current_video_file = filename.name
            print(f"global_instance.current_video_file: {global_instance.current_video_file}")
            global_instance.video_files.append(filename.name)
            with open(global_instance.current_video_file + '.csv', 'a', newline='') as csvfile:
                fieldnames = ['center x', 'center y', 'frame', 'people in frame']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'center x': 'center x', 'center y': 'center y', 'frame': 'frame', 'people in frame': 'people in frame'})

"""
Function that is called at the very end of execution.
Sends generated results in form of csv file to Firebase storage
"""
def send_results_to_firebase():
    bucket = storage.bucket() #reference to Firebase Storage
    current_results = global_instance.current_video_file + ".csv" #path to file that is going to be sent to Storage
    all_results = "movement.csv"
    graph = "graph.png"
    uid = global_instance.uid

    #upload files to Firebase Storage
    destination_file_path = "Results/"  + uid + ".csv"   #send the csv with results to folder Results,  named by User ID (easier access)
    upload_file = bucket.blob(destination_file_path)
    upload_file.upload_from_filename(all_results)
    destination_file_path = "Graphs/"  + uid + ".png"   #send the csv with results to folder Results,  named by User ID (easier access)
    upload_file = bucket.blob(destination_file_path)
    upload_file.upload_from_filename(graph)

    #delete the csv's file from local folder
    if os.path.exists(current_results):
        os.remove(current_results)
        os.remove(all_results)
        os.remove(graph)
        print(f"{current_results} has been deleted.")
        print(f"{all_results} has been deleted.")
        print(f"{graph} has been deleted.")
    else:
        print(f"{current_results} does not exist in the current directory.")

    #delete video from local folder
    local_file_path = "video/" + global_instance.current_video_file
    if os.path.exists(local_file_path):
        print("File exists: ", local_file_path)
        os.remove(local_file_path)
        print(f"{local_file_path} has been deleted.")
    else:
        print(f"{local_file_path} does not exist in the current directory.")
    

def analyze_plot_firebase():
    """Analysis of output results."""
    csv_file = global_instance.current_video_file + ".csv"

    df = pd.read_csv(csv_file) #data frame from csv file using Pandas

    #initialization of variables
    movement_values = []
    grouped = df.groupby('frame')
    one_person = False
    x_array = []
    y_array = []

    #Movement analysis per frame for entire video
    for frame, frame_data in grouped:
        if frame_data.empty: continue
        
        center_x_values = frame_data['center x'].values
        center_y_values = frame_data['center y'].values
        center_x_values = center_x_values[center_x_values != 'center x']
        center_y_values = center_y_values[center_y_values != 'center y']
        ppl = frame_data['people in frame'].values[0]
        if(ppl == "people in frame"): break

        #num_people = int(frame_data['people in frame'].iloc[0])
        num_people = int(ppl)

        if num_people == 1:
            one_person = True
            x_array.append(center_x_values[0])
            y_array.append(center_y_values[0])
            continue

        # Izračunajte srednju vrijednost kretanja za ovaj frame
        total_movement = 0
        for i in range(num_people):
            dx = float(center_x_values[i]) - float(center_x_values[i - 1])
            dy = float(center_y_values[i]) - float(center_y_values[i - 1])
            displacement = np.sqrt(dx**2 + dy**2)
            total_movement += displacement

        # Srednja vrijednost kretanja za ovaj frame
        if num_people > 0:
            average_movement = total_movement / num_people
            movement_values.append(average_movement)
        else:
            movement_values.append(0.0)

    if one_person:
        for i in range(1, len(x_array)):
            dx = float(x_array[i]) - float(x_array[i - 1])
            dy = float(y_array[i]) - float(y_array[i - 1])
            displacement = np.sqrt(dx**2 + dy**2)
            movement_values.append(displacement)
    if len(movement_values) > 0:
        movement = np.array(movement_values).mean()
        print("Nije detektirano kretanje u videu.")
    else:
        movement = 0

    print(f"Movement in this video: {movement}")
    #find movement analysis results of previous videos in Firebase Storage
    bucket = storage.bucket() #reference to Firebase Storage
    folder_path = "Results/"
    blobs = bucket.list_blobs(prefix=folder_path)

    found = False

    for blob in blobs:    
        filename = os.path.basename(blob.name)
        filename = filename.split('.')

        if(filename[0] == global_instance.uid):
            print("Rezultati nadeni")
            blob.download_to_filename("movement.csv")
            print(blob.name)
            found = True
            blob.delete()
            break

    if not found:
        print("Rezultati nisu nadeni")
        with open('movement.csv', 'a', newline='') as csvfile:
            fieldnames = ['movement value', 'date']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'movement value': 'movement value', 'date': 'date'})
            writer.writerow({'movement value': movement, 'date': global_instance.date})

    df = pd.read_csv("movement.csv")
    # Convert the "date" column to strings
    df["date"] = df["date"].astype(str)

    # Filter out rows where "date" is not a valid date
    valid_date_rows = df["date"].str.match(r'\d{4}-\d{2}-\d{2}')
    df = df[valid_date_rows]

    # Convert the "date" column to datetime
    dates = pd.to_datetime(df["date"], format="%Y-%m-%d").to_numpy()


    """valid_date_rows = df["date"].str.match(r'\d{4}-\d{2}-\d{2}')
    df = df[valid_date_rows]
    # Extract data from the DataFrame
    #dates = pd.to_datetime(df["date"]).to_numpy()  # Convert date column to datetime
    # Specify the datetime format
    dates = pd.to_datetime(df["date"], format="%Y-%m-%d").to_numpy()"""
    print(dates)
    movement_values = df["movement value"].to_numpy()
    limit = 5  # Your hardcoded limit value

    # Check if there is already movement value with the same date of recording
    current_date = pd.to_datetime(global_instance.date, format="%Y-%m-%d")
    print(current_date)
    if current_date in dates:
        index = np.where(dates == current_date)[0][0]
        corresponding_movement = movement_values[index]
        movement = (movement + corresponding_movement)/2
        movement_values[index] = movement
    else:
        dates = np.append(dates, current_date)
        movement_values = np.append(movement_values, movement)
        with open('movement.csv', 'a', newline='') as csvfile:
            fieldnames = ['movement value', 'date']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'movement value': 'movement value', 'date': 'date'})
            writer.writerow({'movement value': movement, 'date': global_instance.date})

    print(f"Movement overall: {movement_values}")

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the movement values
    ax.plot(dates, movement_values, label="Vrijednost kretanja", color='blue')

    # Plot the dashed line for the limit
    ax.axhline(limit, color='red', linestyle='--', label="Granica kretanja")

    # Shade the background
    ax.fill_between(dates, movement_values, limit, where=(movement_values > limit), color='green', alpha=0.3)
    ax.fill_between(dates, movement_values, limit, where=(movement_values < limit), color='red', alpha=0.3)

    # Customize the plot
    ax.set_xlabel("Datum")
    ax.set_ylabel("Kretanje")
    ax.set_title("Graf kretanja po datumima")
    ax.grid(True)
    ax.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.savefig("graph.png", format="png")
    
#main method for starting the whole algorithm
if __name__ == "__main__":
    print("Enters predict.py")
    predict()
    for file_name in global_instance.video_files:
        global_instance.current_video_file = file_name
        print(f"global_instance.current_video_file: {global_instance.current_video_file}")
        print(f"Attempting to read file: {file_name}")