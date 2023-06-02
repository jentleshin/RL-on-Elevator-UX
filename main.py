import gym
from gym import spaces
import numpy as np

FLOOR_HIGHT=3.0
MAX_VELOCITY=100.0
DELTA_T=0.1

class ElevatorEnv(gym.Env):
    metadata = {'render.modes': ['human']} # 렌더링 모드
    def __init__(self, render_mode='rgb_array', size=[8,10],floor=5):
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
        self.observation_space = spaces.Dict({"buttonsOut":spaces.MultiBinary(floor),
                            "buttonsIn":spaces.MultiBinary(floor),
                            "location":spaces.Box(low=0,high=FLOOR_HIGHT*floor),
                            "velocity":spaces.Box(low=-MAX_VELOCITY,high=MAX_VELOCITY)})
        self.action_space = spaces.Box(low=-10.0,high=10.0,dtype=np.float32)

    
        self.start_state = dict({"buttonsOut":np.zeros(floor),
                                 "buttonsIn":np.zeros(floor),
                                 "location":0.0,
                                 "velocity":0.0})
        self.terminal_state = size[0] * size[1] - 1
    def probability_function(self,state):
        return False
    def compute_reward(self,state,action,next_state):
        return 0
    def is_done(self,state,action,next_state):
        return self.done
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

        self.observation=next_state
        reward=self.compute_reward(prev_state,action,next_state)
        self.done=self.is_done(prev_state,action,next_state)
             

        
        return (self.observation,reward,self.done)

class PassengerAgent():
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


    def on_board(self,env):
        return


