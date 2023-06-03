from typing import List, Optional, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

FLOOR_HEIGHT=3.0
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
        origin: origin of passenger

        dest: destination of passenger

        on_board: boolean
            whether the passenger is on board the elevator
        arrival: boolean
            whether the passenger arrived the destination
            
        """
        self.origin=origin
        self.dest=dest
        self.on_board=False
        self.arrival=False


class PassengerEnv():
    def __init__(self,passenger_state=[(3,1),(2,1)]):
        self.passengers=list()
        for i in range(len(passenger_state)):
            self.passengers.append(Passenger(passenger_state[i][0],passenger_state[i][1]))
    
    def buttonsPushed(self,tot_floor):
        buttons=np.zeros(tot_floor)
        for i in range(len(self.passengers)):
            buttons[self.passengers[i].origin-1]=1
        return buttons

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
        buttonsIn=list(int)
        for i in range(len(self.passengers)):
            if self.passengers[i].board()==True:
                if self.passengers[i].dest==floor:
                    self.passengers[i].arrival=True
                    self.passengers[i].on_board=False

            if self.passengers[i].origin==floor:
                self.passengers[i].on_board=True
                buttonsIn.append(self.passengers[i].dest)
        
        return buttonsIn

class ElevatorEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, render_mode="rgb_array",tot_floor=3,
                 passenger_status=PassengerEnv(),passenger_mode='determined'):
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
                            "location":spaces.Box(low=0,high=FLOOR_HEIGHT*tot_floor),
                            "velocity":spaces.Box(low=-MAX_VELOCITY,high=MAX_VELOCITY)})
        self.action_space = spaces.Box(low=-10.0,high=10.0,dtype=np.float32)
        self.passenger_status=passenger_status
        self.tot_floor=tot_floor
    
        if passenger_mode=='determined':
            self.start_state = dict({"buttonsOut":passenger_status.buttonsPushed(tot_floor),
                                 "buttonsIn":np.zeros(tot_floor),
                                 "location":0.0,
                                 "velocity":0.0})
        self.terminal_state = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

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
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        elevator_width=50
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
        elevator_height=pix_square_size/2
        floor_pixel_size=(self.window_size-pix_square_size)/(self.tot_floor-1)
        # First we draw the target
        for i in range(self.tot_floor):
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (self.window_size/2,(i+0.5)*pix_square_size),
                pix_square_size / 3,
            )
            if i!=self.tot_floor-1:
                pygame.draw.line(
                    canvas,
                    (0, 0, 255),
                    (self.window_size/2,(i+0.5)*pix_square_size),
                    (self.window_size/2,(i+1.5)*pix_square_size),
                    width=3,
                )
            pygame.draw.circle(
                canvas,
                (255, 255, 255),
                (self.window_size/2,(i+0.5)*pix_square_size),
                pix_square_size / 5,
            )

        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                (self.window_size/2-elevator_width/2,
                 self.window_size-pix_square_size/2-self.observation['location']/FLOOR_HEIGHT*floor_pixel_size-elevator_height),
                (elevator_width, elevator_height),
            ),
        )

        # Now we draw the agent
        for i in range(len(self.passenger_status.passengers)):
            if self.passenger_status.passengers[i].on_board is False:
                pygame.draw.circle(
                    canvas,
                    (0, 0, 255),
                    (self.window_size/2 -50 ,(i+0.5)*pix_square_size ),
                    pix_square_size / 5,
                )
            
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
    
    def reset(self):
    # 환경 초기화
        self.done=False
        self.observation=self.start_state
        return self.observation
        
    def step(self, action):
    # 주어진 동작을 위치로 마크를 배치
        buttonsIn=None
        prev_state=self.observation
        next_state=prev_state.copy()

        next_state['location']+=next_state['velocity']*DELTA_T+0.5*action*pow(DELTA_T,2)
        next_state['velocity']+=DELTA_T*action

        curr_floor=self.check_floor(prev_state,action,next_state)
        if curr_floor!=-1:
            buttonsIn=self.passenger_status.on_off_board(curr_floor)
            if buttonsIn!=None:
                for i in range(len(buttonsIn)):
                    next_state['buttonsOut'][buttonsIn[i]]=0
                    next_state['buttonsIn'][buttonsIn[i]]=1

        reward=self.compute_reward(prev_state,action,next_state)

        self.observation=next_state        
        self.done=self.is_done(prev_state,action,next_state)
                
        return (self.observation,reward,self.done)
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()



