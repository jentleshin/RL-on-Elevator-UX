import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from enum import Enum
FLOOR_HEIGHT=3.0

class MovingState(Enum):
    UP = 0
    DOWN = 1
    STOP = 2

class baseline_policy():
    def __init__(self,max_accel,tot_floor):
        self.moving_direction=MovingState.STOP
        self.destinations=list()
        self.max_accel=max_accel
        self.tot_floor=tot_floor

    def action(self,curr_state):
        location=curr_state["location"][0]
        velocity=curr_state["velocity"][0]
        buttons=np.logical_or( curr_state["buttonsOut"],curr_state["buttonsIn"])



    
    def add_destination(self,location,velocity,buttons):
        if self.moving_direction==MovingState.STOP:
            floor=location/FLOOR_HEIGHT
            self.moving_direction=self.select_direction(floor,buttons)
            available_dest=self.available_destination(self,location,velocity)
            dest=[i for i in available_dest 
                  if buttons[i]==1]
            return dest
        elif self.moving_direction==MovingState.UP:
            



    def select_direction(self,floor,buttons):
        for i in range(1,min(floor,len(buttons)-floor-1)):
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
        
        

        


    



