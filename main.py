from typing import List, Optional, Union
import gym
from gym import spaces
import numpy as np

FLOOR_HIGHT=3.0
MAX_VELOCITY=100.0
DELTA_T=0.1
STEP_REWARD=-0.1
FLOOR_RANGE=0.1
STOP_VEL_RANGE=0.1
REWARD_SUCCESS=10
ACCEL_THRESHOLD=5

class Passenger():
    def __init__(self, origin,dest):
        """
        An initialization function

        Parameters
        ----------
        size: a list of integers
            the dimension of 2D grid environment
        start: integer
            start state (i.e., location)

        """
        self.origin=origin
        self.dest=dest
        self.on_board=False
        self.arrival=False



class PassengerEnv():
    def __init__(self,passenger_state=[(3,1),(2,1)]):
        self.passengers=list(Passenger)
        for i in range(len(passenger_state)):
            self.passengers.append(Passenger(passenger_state[i][0],passenger_state[i][1]))
    
    def num_on_board(self):
        num=0
        for i in range(len(self.passengers)):
            if self.passengers[i].board() is True:
                num+=1
        return num
    
    def check_arrival(self):
        num=0
        i=0
        while True:
            if not self.passengers:
                break
            if i>len(self.passengers):
                break
            if self.passengers[i].arrival is True:
                num+=1
                self.passengers.pop(i)
            else:
                i+=1
        return num
    
    def on_off_board(self,floor):
        for i in range(len(self.passengers)):
            if self.passengers[i].board()==True:
                if self.passengers[i].dest==floor:
                    self.passengers[i].arrival=True
                    self.passengers[i].on_board=False
            if self.passengers[i].origin==floor:
                self.passengers[i].on_board=True

class ElevatorEnv(gym.Env):
    metadata = {'render.modes': ['human']} # 렌더링 모드
    def __init__(self, render_mode='rgb_array',tot_floor=3,passenger_status=PassengerEnv()):
        """
        An initialization function

        Parameters
        ----------
        size: a list of integers
            the dimension of 2D grid environment
        start: integer
            start state (i.e., location)
        epsilon: float
            the probability of taking random actions
        obstacle: 

        """
        self.render_mode = render_mode

        ''' set observation space and action space '''
        self.observation_space = spaces.Dict({"buttonsOut":spaces.MultiBinary(tot_floor),
                            "buttonsIn":spaces.MultiBinary(tot_floor),
                            "location":spaces.Box(low=0,high=FLOOR_HIGHT*tot_floor),
                            "velocity":spaces.Box(low=-MAX_VELOCITY,high=MAX_VELOCITY)})
        self.action_space = spaces.Box(low=-10.0,high=10.0,dtype=np.float32)
        self.passenger_status=passenger_status
        self.tot_floor=tot_floor
    
        self.start_state = dict({"buttonsOut":np.zeros(tot_floor),
                                 "buttonsIn":np.zeros(tot_floor),
                                 "location":0.0,
                                 "velocity":0.0})
        self.terminal_state = 0

    def probability_function(self,state):
        return False
    
    def check_floor(self,state,action,next_state):
        curr_floor=-1
        for i in range(self.tot_floor):
            if abs(i-next_state['location'])<FLOOR_RANGE:
                curr_floor=i
        if curr_floor==-1:
            return -1
        if abs(next_state['velocity'])<STOP_VEL_RANGE:
            return curr_floor
        else:
            return -1

    def accel_relu(self,action):
        if abs(action)>ACCEL_THRESHOLD:
            return ACCEL_THRESHOLD-abs(action)
        else:
            return 0
        
    def compute_reward(self,state,action,next_state):
        reward=self.accel_relu(action)
        reward+=STEP_REWARD+self.passenger_status.check_arrival()*REWARD_SUCCESS
        return reward
    
    def is_done(self,state,action,next_state):
        if self.passenger_status.passengers:
            self.done=True
        return self.done
    
    def render(self):
        return super().render()
    
    def reset(self):
    # 환경 초기화
        self.done=False
        self.observation=self.start_state
        return self.observation
        
    def step(self, action):
    # 주어진 동작을 위치로 마크를 배치
        prev_state=self.observation
        next_state=prev_state.copy()

        next_state['location']+=next_state['velocity']*DELTA_T+0.5*action*pow(DELTA_T,2)
        next_state['velocity']+=DELTA_T*action

        curr_floor=self.check_floor(prev_state,action,next_state)
        if curr_floor!=-1:
            self.passenger_status.on_off_board(curr_floor)

        reward=self.compute_reward(prev_state,action,next_state)

        self.observation=next_state        
        self.done=self.is_done(prev_state,action,next_state)
                
        return (self.observation,reward,self.done)




