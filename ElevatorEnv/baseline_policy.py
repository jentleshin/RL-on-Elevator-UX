import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from enum import Enum
FLOOR_HEIGHT=3.0
DELTA_T=0.1
STOP_VEL_RANGE=0.1
ACCEL_THRESHOLD=1.0
FLOOR_RANGE=0.1

class MovingState(Enum):
    UP = 0
    DOWN = 1
    STOP = 2

class baseline_policy():
    def __init__(self,max_accel=ACCEL_THRESHOLD,tot_floor=5):
        self.moving_direction=MovingState.STOP
        self.destinations=list()
        self.max_accel=max_accel
        self.tot_floor=tot_floor
        self.dest=[]
        self.T=0

    def action(self,curr_state):
        location=curr_state["location"][0]
        velocity=curr_state["velocity"][0]
        #print("loc: {},vel: {}".format(location,velocity))
        #print(self.moving_direction)
        if abs(velocity)<STOP_VEL_RANGE and not self.dest:
            self.moving_direction=MovingState.STOP
        buttons=np.logical_or( curr_state["buttonsOut"],curr_state["buttonsIn"])
        self.add_destination(location,velocity,buttons)
        if self.dest is None or not self.dest:
            return np.array([0.0], dtype=np.float32)
        else:
            self.T+=DELTA_T
            if abs(velocity)<STOP_VEL_RANGE and abs(location-self.dest[0]*FLOOR_HEIGHT)<FLOOR_RANGE:
                self.dest.remove(self.dest[0])
                self.T=0
                return np.array([0.0], dtype=np.float32)

            distance=abs(location-self.dest[0]*FLOOR_HEIGHT)
            #print("dist: {:.2f}".format(distance))
            if distance<=0.5*self.max_accel*np.square(DELTA_T):# and distance>=np.square(velocity)/2.0/self.max_accel:
                accel=-abs(velocity)/DELTA_T
            elif distance<=self.max_accel*np.square(DELTA_T):
                accel=-abs(velocity)/2.0/DELTA_T
            elif distance<(self.max_accel*np.square(DELTA_T)+2*abs(velocity)*DELTA_T+np.square(velocity)/2.0/self.max_accel):
                #print("{:.2f}".format(self.max_accel*np.square(DELTA_T)+2*abs(velocity)*DELTA_T+np.square(velocity)/2.0/self.max_accel))
                accel=-self.max_accel
            else:
                accel=self.max_accel

            
        if self.moving_direction == MovingState.UP:
            accel=accel
        elif self.moving_direction==MovingState.DOWN:
            accel=-accel
        #print("accel: {}".format(accel))
        return np.array([accel], dtype=np.float32)
    
    def add_destination(self,location,velocity,buttons):
        if self.moving_direction==MovingState.STOP:
            floor=int(np.round(location/FLOOR_HEIGHT))
            self.moving_direction=self.select_direction(floor,buttons)
            if self.moving_direction==MovingState.STOP:
                return None
        available_dest=self.available_destination(location,velocity)
        if len(available_dest)==0:return
        for i in available_dest:
            if buttons[i]==True and i not in self.dest:
                self.dest.append(i)
        if self.moving_direction==MovingState.UP:
            self.dest.sort()
        elif self.moving_direction==MovingState.DOWN:
            self.dest.sort(reverse=True)

    
    def select_direction(self,floor,buttons):
        for i in range(0,max(floor+1,len(buttons)-floor)):
            if floor+i<len(buttons):
                if buttons[floor+i]==1:
                    if i==0:
                        return MovingState.STOP
                    return MovingState.UP
            if floor-i>=0:
                if buttons[floor-i]==1:
                    return MovingState.DOWN
        return MovingState.STOP

    #def destination(self,curr_state):

    def available_destination(self,location,velocity):
        min_distance=np.square(velocity)/2.0/self.max_accel
        if self.moving_direction==MovingState.UP:
            if location>(self.tot_floor-1)*FLOOR_HEIGHT:return list()
            return np.arange(np.ceil((location+min_distance)/FLOOR_HEIGHT),self.tot_floor,dtype=np.int16)
        elif self.moving_direction==MovingState.DOWN:
            if location<0:return list()
            return np.arange(np.floor((location-min_distance)/FLOOR_HEIGHT),-1,-1,dtype=np.int16)
        

        


    



