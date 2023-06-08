import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from enum import Enum
FLOOR_HEIGHT=3.0
DELTA_T=0.1
STOP_VEL_RANGE=0.1
ACCEL_THRESHOLD=5

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

    def action(self,curr_state):
        location=curr_state["location"][0]
        velocity=curr_state["velocity"][0]
        if abs(velocity)<STOP_VEL_RANGE:
            self.moving_direction=MovingState.STOP
        buttons=np.logical_or( curr_state["buttonsOut"],curr_state["buttonsIn"])
        dest=self.add_destination(location,velocity,buttons)
        if dest is None:
            return np.array([0.0], dtype=np.float32)
        else:
            distance=abs(location-dest[0]*FLOOR_HEIGHT)
            if distance<abs(velocity)*DELTA_T-0.5*self.max_accel*np.square(DELTA_T):
                accel=(distance-abs(velocity)*DELTA_T)*2.0/np.square(DELTA_T)
            elif distance<np.square(velocity)/2.0/self.max_accel:
                accel=-self.max_accel
            else:
                accel=self.max_accel
            
        if self.moving_direction == MovingState.UP:
            accel=accel
        elif self.moving_direction==MovingState.DOWN:
            accel=-accel
        return np.array([accel], dtype=np.float32)
    
    def add_destination(self,location,velocity,buttons):
        if self.moving_direction==MovingState.STOP:
            floor=int(location/FLOOR_HEIGHT)
            self.moving_direction=self.select_direction(floor,buttons)
            if self.moving_direction==MovingState.STOP:
                return None
        available_dest=self.available_destination(location,velocity)     
        dest=[i for i in available_dest 
                if buttons[i]==1]
        return dest
    
    def select_direction(self,floor,buttons):
        for i in range(1,min(floor+1,len(buttons)-floor)):
            if buttons[floor+i]==1:
                return MovingState.UP
            elif buttons[floor-i]==1:
                return MovingState.DOWN
        return MovingState.STOP

    #def destination(self,curr_state):

    def available_destination(self,location,velocity):
        min_distance=np.square(velocity)/2.0/self.max_accel
        if self.moving_direction==MovingState.UP:
            return np.arange(np.ceil((location+min_distance)/FLOOR_HEIGHT),self.tot_floor)
        elif self.moving_direction==MovingState.DOWN:
            return np.arange(np.floor((location-min_distance)/FLOOR_HEIGHT),-1)
        

        


    



