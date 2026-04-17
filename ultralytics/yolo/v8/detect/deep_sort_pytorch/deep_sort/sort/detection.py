# vim: expandtab:ts=4:sw=4
import numpy as np
import csv
import datetime

import os

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import globals
from globals import Globals


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    global_instance = Globals() #global variables are accessed through this variable


    def __init__(self, tlwh, confidence, feature, oid):
        """Initialize the variables to their initial values (provided when
        creating the instance of this class).
        """
        self.tlwh = np.asarray(tlwh, dtype=float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.oid = oid
        

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        Call method to save that format to output.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        
        self.save_movement_to_csv(ret, Detection.global_instance.current_video_file + '.csv', Detection.global_instance.get_global_frame(), Detection.global_instance.get_no_of_people())

        return ret


    def save_movement_to_csv(self, ret, filename, frame, no_of_people):
        """Save bounding box format (center x, center y, aspect ratio,
        height) and current number of detection of people to output csv
        file for later movement analysis. Fieldnames define which variable
        is written in which column.
        """
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = ['center x', 'center y', 'frame', 'people in frame']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'center x': ret[0], 'center y': ret[1],  'frame': frame, 'people in frame': no_of_people})
