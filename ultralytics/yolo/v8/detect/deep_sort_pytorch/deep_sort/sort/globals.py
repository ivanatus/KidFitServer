# -*- coding: utf-8 -*-
"""
The class for keeping current values of global variables.
The variable values are changed for each frame of the video.
There are getter and setter functions.
Getter functions (get_variable()) return current value of global variable.
Setter functions (set_variable(value)) set global variable to specified value.
Variables that are kept in this class: current frame of the video, number of people in the current frame of the video,
uid of user who's video is currently being analyzed and date of the video upload to Firebase.
"""

class Globals:
    
    global_frame = 0
    no_of_people = 0
    video_files = []
    current_video_file = ""
    uid = ""
    date = ""
    
    #sets value of global frame variable to value of current frame of video that is being analyzed
    @staticmethod
    def set_global_frame(frame):
        Globals.global_frame = frame
        
    #returns value of current frame of the video that is being analyzed
    @staticmethod
    def get_global_frame():
        return Globals.global_frame
    
    #sets value of global variable that represents number of detected people in the current frame of the video
    @staticmethod
    def set_no_of_people(n):
        Globals.no_of_people = n
        
    #returns value of variable that represents number of detected people in the current frame of the video
    @staticmethod
    def get_no_of_people():
        return Globals.no_of_people
    
    #sets global variable that represents uid of user that uploaded the currently analyzed video
    @staticmethod
    def set_uid(m):
        Globals.uid = m
        print("uid ", Globals.uid)
        
    #returns value of variable that represents uid of user that uploaded the currently analyzed video
    @staticmethod
    def get_uid():
        return Globals.uid
    
    #sets value of date of user's recording of current video
    @staticmethod
    def set_date(m):
        Globals.date = m
        print("date ", Globals.date)
        
    #returns value of date of user's recording of current video
    @staticmethod
    def get_date():
        return Globals.date