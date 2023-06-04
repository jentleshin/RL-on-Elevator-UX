from typing import List, Optional, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from enum import Enum

FLOOR_HEIGHT=3.0
MAX_VELOCITY=100.0
DELTA_T=0.1
STEP_REWARD=-0.1
FLOOR_RANGE=0.1
STOP_VEL_RANGE=0.1
REWARD_SUCCESS=10
ACCEL_THRESHOLD=5

class State(Enum):
    WAIT = 0
    ONBOARD = 1
    ARRIVAL = 2

class Passenger():
    def __init__(self, origin, dest):
        """
        An initialization function

        Parameters
        ----------
        origin: origin of passenger

        dest: destination of passenger

        state: one of WAIT, ONBOARD, ARRIVAL
            
        """
        self.origin=origin
        self.dest=dest
        self.state = None

class PassengerEnv():
    def __init__(self, tot_floor, passenger_args=[(2,0),(1,0)]):
        self.passengers=list()
        self.tot_floor = tot_floor
        self.passenger_args = passenger_args

        self.waiting = {i: [] for i in range(tot_floor)}
        self.onboarding = {i: [] for i in range(tot_floor)}
        self.arrived = {i: [] for i in range(tot_floor)}
        
        # reward_args used for calculating reward
        self.current_arrival = 0

        for passenger_arg in self.passenger_args:
            self.create(passenger_arg)
    
    def create(self, passenger_args):
        passenger = Passenger(*passenger_args)
        passenger.state = State.WAIT
        self.waiting[passenger.origin].append(passenger)
        self.passengers.append(passenger)

    def get_buttonsIn(self):
        return np.array([bool(value) for value in self.onboarding.values()])
    
    def get_buttonsOut(self):
        return np.array([bool(value) for value in self.waiting.values()])
    
    def on_off_board(self,floor):
        for passenger in self.waiting[floor]:
            passenger.state = State.ONBOARD
            self.onboarding[passenger.dest].append(passenger)
        self.waiting[floor]=[]

        self.num_arrival = len(self.onboarding[floor])
        for passenger in self.onboarding[floor]:
            passenger.state = State.ARRIVAL
            self.arrived[floor].append(passenger)
        self.onboarding[floor]=[]
        return
    
    def get_current_arrival(self):
        return self.current_arrival
    
    def reset_current_arrival(self):
        self.current_arrival = 0
        return
    
    def all_arrived(self):
        return all(passenger.state==State.ARRIVAL for passenger in self.passengers)
    
    def reset(self):
        self.__init__(self.tot_floor, self.passenger_args)
    
class ElevatorEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, render_mode="rgb_array",tot_floor=3, passenger_mode='determined'):
        self.render_mode = render_mode

        ''' set observation space and action space '''
        self.observation_space = spaces.Dict({
            "buttonsOut":spaces.MultiBinary(tot_floor),
            "buttonsIn":spaces.MultiBinary(tot_floor),
            "location":spaces.Box(low=0,high=FLOOR_HEIGHT*tot_floor),
            "velocity":spaces.Box(low=-MAX_VELOCITY,high=MAX_VELOCITY)
            })
        self.action_space = spaces.Box(low=-10.0,high=10.0,dtype=np.float32)
        
        self.tot_floor=tot_floor
        self.passengerEnv=PassengerEnv(self.tot_floor)
    
        if passenger_mode=='determined':
            self.start_state = dict({"buttonsOut":self.passengerEnv.get_buttonsOut(),
                                     "buttonsIn":self.passengerEnv.get_buttonsIn(),
                                     "location": np.array([0.0], dtype=np.float32),
                                     "velocity": np.array([0.0], dtype=np.float32)
                                     })
        self.terminal_state = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        pygame.font.init()
        self.font=pygame.font.Font('freesansbold.ttf',100)

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window_size=900
        self.window = None
        self.clock = None


    def probability_function(self,state):
        return False
    
    def check_floor(self, location, velocity):
        location = location[0] / FLOOR_HEIGHT
        velocity = velocity[0]
        if abs(round(location)-location)<FLOOR_RANGE and abs(velocity)<STOP_VEL_RANGE:
            return True, round(location)
        else:
            return False, None
        
    def clip(self, location, velocity):
        if  location[0]<0:
            location=np.array([0.0], dtype=np.float32)
            velocity=np.array([0.0], dtype=np.float32)
        elif location[0]>FLOOR_HEIGHT*(self.tot_floor-1):
            location=np.array([FLOOR_HEIGHT * (self.tot_floor - 1)], dtype=np.float32)
            velocity=np.array([0.0], dtype=np.float32)
        return location, velocity
    
    def accel_relu(self,action):
        if abs(action)>ACCEL_THRESHOLD:
            return ACCEL_THRESHOLD-abs(action)
        else:
            return 0
        
    def compute_reward(self, prev_state, action, next_state):
        reward_arg = self.passengerEnv.get_current_arrival()
        reward=self.accel_relu(action)+STEP_REWARD+(reward_arg)*REWARD_SUCCESS
        self.passengerEnv.reset_current_arrival()
        return reward
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        color=[(0,0,0),(0,255,0)]
        elevator_width=40
        elevator_height=0
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.tot_floor
        )  # The size of a single grid square in pixels
        elevator_height=pix_square_size/3
        floor_pixel_size=(self.window_size-pix_square_size)/(self.tot_floor-1)
        # First we draw the target
        for i in range(self.tot_floor):
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (self.window_size/2,(i+0.5)*pix_square_size),
                40,
            )
            if i!=self.tot_floor-1:
                pygame.draw.line(
                    canvas,
                    (0, 0, 255),
                    (self.window_size/2,(i+0.5)*pix_square_size),
                    (self.window_size/2,(i+1.5)*pix_square_size),
                    width=10,
                )
            pygame.draw.circle(
                canvas,
                (255, 255, 255),
                (self.window_size/2,(i+0.5)*pix_square_size),
                20,
            )
            pygame.draw.circle(
                canvas,
                color[self.observation['buttonsIn'][i]],
                (self.window_size -100,self.window_size-(i+0.5)*pix_square_size),
                20,
            )
            pygame.draw.circle(
                canvas,
                color[self.observation['buttonsOut'][i]],
                (100,self.window_size-(i+0.5)*pix_square_size),
                20,
            )

        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                (self.window_size/2-elevator_width/2,self.window_size-pix_square_size/2-self.observation['location'][0]/FLOOR_HEIGHT*floor_pixel_size-elevator_height),
                (elevator_width, elevator_height),
            ),
        )
        num_arrived=0
        # Now we draw the agent
        for i in range(len(self.passengerEnv.passengers)):
            if self.passengerEnv.passengers[i].state is State.ARRIVAL:
                pygame.draw.circle(
                    canvas,
                    (255, 0, 0),
                    (self.window_size/2 -100 - 50*num_arrived ,self.window_size-(0.5)*pix_square_size ),
                    20,
                )
            elif self.passengerEnv.passengers[i].state is State.WAIT:
                pygame.draw.circle(
                    canvas,
                    (0, 0, 255),
                    (self.window_size/2 -100 ,(i+0.5)*pix_square_size ),
                    20,
                )

        text=self.font.render(str(round(self.observation['location'][0],2)),True,(0,0,0),(255,255,255))
        textRect=text.get_rect()
        textRect.center=(150,80)
        canvas.blit(text,textRect)

            
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def reset(self, seed=None, options=None):
    # 환경 초기화
        self.done=False
        self.passengerEnv.reset()
        self.observation=self.start_state
        return self.observation,{"info":None}
        
    def step(self, action):
    # 주어진 동작을 위치로 마크를 배치
    # elevator 1층, 최상층 범위 벗어나려 할때 reward needed
        truncated=False
        
        prev_state=self.observation
        next_state=prev_state.copy()
        next_state["location"]+=DELTA_T*next_state['velocity']+0.5*action*pow(DELTA_T,2)
        next_state['velocity']+=DELTA_T*action
        next_state["location"], next_state["velocity"] = self.clip(next_state["location"], next_state["velocity"])
        
        is_floor,floor=self.check_floor(next_state["location"], next_state["velocity"])
        if is_floor:
            self.passengerEnv.on_off_board(floor)
            next_state['buttonsOut']=self.passengerEnv.get_buttonsOut()
            next_state['buttonsIn']=self.passengerEnv.get_buttonsIn()
        self.observation=next_state        

        reward=self.compute_reward(prev_state,action,next_state)
        self.done = self.passengerEnv.all_arrived()

        return self.observation,reward,self.done,truncated,{"info":None}
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
