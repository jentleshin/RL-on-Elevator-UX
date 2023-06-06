import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from enum import Enum
import copy

FLOOR_HEIGHT=3.0
MAX_VELOCITY=100.0
MIN_VELOCITY=-MAX_VELOCITY
DELTA_T=0.1

FLOOR_RANGE=0.2
EDGE_FLOOR_RANGE=1
STOP_VEL_RANGE=0.1
ACCEL_THRESHOLD=5

STEP_REWARD=-0.1
ACCEL_REWARD=-1
CURRIVAL_REWARD=10

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

        current_arrival = len(self.onboarding[floor])
        for passenger in self.onboarding[floor]:
            passenger.state = State.ARRIVAL
            self.arrived[floor].append(passenger)
        self.onboarding[floor]=[]
        return current_arrival
    
    def all_arrived(self):
        return all(passenger.state==State.ARRIVAL for passenger in self.passengers)
    
    def reset(self):
        self.__init__(self.tot_floor, self.passenger_args)
    
class ElevatorEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4, "passenger_mode":["determined","randomly_fixed","random"]}
    def __init__(self, render_mode="rgb_array",tot_floor=5, passenger_mode="random",passenger_num=5):
        self.render_mode = render_mode

        ''' set observation space and action space '''
        self.MIN_LOCATION = FLOOR_HEIGHT*(-EDGE_FLOOR_RANGE)
        self.MAX_LOCATION = FLOOR_HEIGHT*(tot_floor-1+EDGE_FLOOR_RANGE)

        self.observation_space = spaces.Dict({
            "buttonsOut":spaces.MultiBinary(tot_floor),
            "buttonsIn":spaces.MultiBinary(tot_floor),
            "location":spaces.Box(low=self.MIN_LOCATION,high=self.MAX_LOCATION),
            "velocity":spaces.Box(low=MIN_VELOCITY,high=MAX_VELOCITY)
            })
        self.action_space = spaces.Box(low=-10.0,high=10.0,dtype=np.float32)
        self.reward=0
        
        self.tot_floor=tot_floor
        self.passenger_mode=passenger_mode
        self.passenger_num=passenger_num
        
    
        if passenger_mode=="determined":
            self.passengerEnv=PassengerEnv(self.tot_floor)
        
        elif "random" in passenger_mode:
            passenger_args=self.randomly_fix_passenger_args(tot_floor,passenger_num)
            self.passengerEnv=PassengerEnv(self.tot_floor,passenger_args)

        self.start_state = dict({"buttonsOut":self.passengerEnv.get_buttonsOut(),
                                     "buttonsIn":self.passengerEnv.get_buttonsIn(),
                                     "location": np.array([np.random.rand()*(self.tot_floor-1)*FLOOR_HEIGHT], dtype=np.float32),
                                     "velocity": np.array([0.0], dtype=np.float32)
                                     })

        self.terminal_state = 0
        self.reward_args = {
            "accel_overload":0,
            "current_arrival":0
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        pygame.font.init()
        self.font=pygame.font.Font('freesansbold.ttf',40)
        self.numfont=pygame.font.Font('freesansbold.ttf',30)

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

    def randomly_fix_passenger_args(self,tot_floor,passenger_num):
        passenger_args=[]

        for i in range(passenger_num):
            passenger_origin=np.random.randint(0,tot_floor)
            passenger_dest=np.random.randint(0,tot_floor)
            while passenger_dest==passenger_origin:
                passenger_dest=np.random.randint(0,tot_floor)
            passenger_args.append((passenger_origin,passenger_dest))
        return passenger_args

    def probability_function(self,state):
        return False
    
    def check_floor(self, location, velocity):
        location = location[0] / FLOOR_HEIGHT
        velocity = velocity[0]
        if 0<=round(location) and round(location)<self.tot_floor and abs(round(location)-location)<FLOOR_RANGE and abs(velocity)<STOP_VEL_RANGE:
            return True, round(location)
        else:
            return False, None
        
    def on_off_board(self, floor):
        current_arrival = self.passengerEnv.on_off_board(floor)
        self.reward_args["current_arrival"] = current_arrival
        return

    def clip(self, state):
        if  state["location"][0]<self.MIN_LOCATION:
            state["location"]=np.array([self.MIN_LOCATION], dtype=np.float32)
            state["velocity"]=np.array([0.0], dtype=np.float32)
        
        elif state["location"][0]>self.MAX_LOCATION:
            state["location"]=np.array([self.MAX_LOCATION], dtype=np.float32)
            state["velocity"]=np.array([0.0], dtype=np.float32)
        return
    
    def accel_relu(self,action):
        if abs(action[0])>ACCEL_THRESHOLD:
            return abs(action[0])-ACCEL_THRESHOLD
        else:
            return 0
        
    def compute_reward(self, prev_state, action, next_state):
        self.reward_args["accel_overload"] = self.accel_relu(action)
        accel_overLoad, current_arrival = self.reward_args.values()
        reward = STEP_REWARD+(accel_overLoad)*ACCEL_REWARD+(current_arrival)*CURRIVAL_REWARD
        self.reward_args = {key: 0 for key in self.reward_args}
        #print(f"{STEP_REWARD}, {(accel_overLoad)*ACCEL_REWARD}, {(bump_velocity)*BUMP_REWARD}, {(current_arrival)*CURRIVAL_REWARD}")
        return reward
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        color=[(0,0,0),(0,255,0)]

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
        elevator_width=elevator_height/2

        floor_pixel_size=(self.window_size-pix_square_size)/(self.tot_floor-1)
        borderLine_thickness=floor_pixel_size * FLOOR_RANGE
        # First we draw the target
        for i in range(self.tot_floor):
            pygame.draw.rect(
                canvas,
                (0,0,255),
                pygame.Rect(
                    (self.window_size/2-(elevator_width+2*borderLine_thickness)/2,(i+0.5)*pix_square_size-(elevator_height+2*borderLine_thickness)/2),
                    (elevator_width+2*borderLine_thickness,elevator_height+2*borderLine_thickness)
                )
            )
            # pygame.draw.circle(
            #     canvas,
            #     (0, 0, 255),
            #     (self.window_size/2,(i+0.5)*pix_square_size),
            #     40,
            # )
            if i!=self.tot_floor-1:
                pygame.draw.line(
                    canvas,
                    (0, 0, 255),
                    (self.window_size/2,(i+0.5)*pix_square_size),
                    (self.window_size/2,(i+1.5)*pix_square_size),
                    width=10,
                )
            pygame.draw.rect(
                canvas,
                (255,255,255),
                pygame.Rect(
                    (self.window_size/2-elevator_width/2,(i+0.5)*pix_square_size-elevator_height/2),
                    (elevator_width,elevator_height)
                )
            )

            # pygame.draw.circle(
            #     canvas,
            #     (255, 255, 255),
            #     (self.window_size/2,(i+0.5)*pix_square_size),
            #     20,
            # )
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
                (self.window_size/2-elevator_width/2,self.window_size-pix_square_size/2-self.observation['location'][0]/FLOOR_HEIGHT*floor_pixel_size-elevator_height/2),
                (elevator_width, elevator_height),
            ),
        )
        num_arrived=np.zeros(self.tot_floor)
        num_waiting=np.zeros(self.tot_floor)

        # Now we draw the agent
        for i in range(len(self.passengerEnv.passengers)):
            if self.passengerEnv.passengers[i].state is State.ARRIVAL:
                pygame.draw.circle(
                    canvas,
                    (255, 0, 0),
                    (self.window_size/2 +100 + 50*num_arrived[self.passengerEnv.passengers[i].dest] ,self.window_size-(self.passengerEnv.passengers[i].dest+0.5)*pix_square_size),
                    20,
                )
                origin_text=self.numfont.render(str(self.passengerEnv.passengers[i].origin) ,True,(255,255,255))
                origin_text_rect=origin_text.get_rect()
                origin_text_rect.center=(self.window_size/2 +100 + 50*num_arrived[self.passengerEnv.passengers[i].dest],self.window_size-(self.passengerEnv.passengers[i].dest+0.5)*pix_square_size )
                canvas.blit(origin_text,origin_text_rect)
                num_arrived[self.passengerEnv.passengers[i].dest]+=1
            elif self.passengerEnv.passengers[i].state is State.WAIT:
                pygame.draw.circle(
                    canvas,
                    (0, 0, 255),
                    (self.window_size/2 -100- 50*num_waiting[self.passengerEnv.passengers[i].origin] ,self.window_size-(self.passengerEnv.passengers[i].origin+0.5)*pix_square_size ),
                    20,
                )
                dest_text=self.numfont.render(str(self.passengerEnv.passengers[i].dest),True,(255,255,255))
                dest_text_rect=dest_text.get_rect()
                dest_text_rect.center=(self.window_size/2 -100 -50*num_waiting[self.passengerEnv.passengers[i].origin] ,self.window_size-(self.passengerEnv.passengers[i].origin+0.5)*pix_square_size )
                canvas.blit(dest_text,dest_text_rect)
                num_waiting[self.passengerEnv.passengers[i].origin]+=1

        loc_text=self.font.render("loc(m): "+str(round(self.observation['location'][0],2)),True,(0,0,0))
        loc_text_rect=loc_text.get_rect()
        loc_text_rect.center=(150,40)
        canvas.blit(loc_text,loc_text_rect)

        rew_text=self.font.render("rew : "+str(round(self.reward,2)),True,(0,0,0))
        rew_text_rect=rew_text.get_rect()
        rew_text_rect.center=(800,40)
        canvas.blit(rew_text,rew_text_rect)

            
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

        self.observation['location']=np.array([np.random.rand()*(self.tot_floor-1)*FLOOR_HEIGHT], dtype=np.float32)
        self.reward=0
        if self.passenger_mode=="random":
            passenger_args=self.randomly_fix_passenger_args(self.tot_floor,self.passenger_num)
            self.passengerEnv=PassengerEnv(self.tot_floor,passenger_args)
            self.observation['buttonsOut']=self.passengerEnv.get_buttonsOut()
            self.observation['buttonsIn']=self.passengerEnv.get_buttonsIn()
        return self.observation,{"info":None}
        
    def step(self, action):
    # 주어진 동작을 위치로 마크를 배치
    # elevator 1층, 최상층 범위 벗어나려 할때 reward needed
        truncated=False
        
        prev_state=self.observation
        next_state=copy.deepcopy(prev_state)
        next_state["location"]+=DELTA_T*next_state['velocity']+0.5*action*pow(DELTA_T,2)
        next_state['velocity']+=DELTA_T*action
        self.clip(next_state)
        
        is_floor,floor=self.check_floor(next_state["location"], next_state["velocity"])
        if is_floor:
            self.on_off_board(floor)
            next_state['buttonsOut']=self.passengerEnv.get_buttonsOut()
            next_state['buttonsIn']=self.passengerEnv.get_buttonsIn()
        self.observation=next_state        

        reward=self.compute_reward(prev_state,action,next_state)
        self.reward+=reward
        self.done = self.passengerEnv.all_arrived()

        return self.observation,reward,self.done,truncated,{"info":None}
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
