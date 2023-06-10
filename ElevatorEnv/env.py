import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from enum import Enum

FLOOR_HEIGHT=3.0
MAX_VELOCITY=100.0
MIN_VELOCITY=-MAX_VELOCITY
DELTA_T=0.1

FLOOR_RANGE=0.1
EDGE_FLOOR_RANGE=1
STOP_VEL_RANGE=0.1
ACCEL_THRESHOLD=1.0

STEP_REWARD=-0.1
ACCEL_REWARD=-1
NONSTOP_REWARD=-1
CURRIVAL_REWARD=10
WAITING_TIME_REWARD=-0.1
DELAYED_TIME_REWARD=-0.1

ZERO_FLOOR_DISTRIBUTION_FACTOR=0.1
NORMAL_FLOOR_DISTRIBUTION_FACTOR=0.05

class State(Enum):
    WAIT = 0
    ONBOARD = 1
    ARRIVAL = 2

class Passenger():
    def __init__(self, origin, dest,creation_time):
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
        self.creation_time=creation_time
        self.onboarding_time=None
        self.expected_arrival_time=self.normal_elevator_arrival_baseline()
    
    def normal_elevator_arrival_baseline(self):
        distance=abs(self.origin-self.dest)*FLOOR_HEIGHT
        return np.sqrt(4*distance/ACCEL_THRESHOLD)

class PassengerEnv():
    def __init__(self, tot_floor, passenger_args=[(2,0,0.0),(1,0,0.0),(3,0,0.0),(4,0,0.0)]):
        self.passengers=list()
        self.tot_floor = tot_floor
        self.passenger_args = passenger_args

        self.waiting = {i: [] for i in range(tot_floor)}
        self.onboarding = {i: [] for i in range(tot_floor)}
        self.arrived = {i: [] for i in range(tot_floor)}
        
        self.create(passenger_args)
    
    def create(self, passenger_args):
        for passenger_arg in passenger_args:
            passenger = Passenger(*passenger_arg)
            passenger.state = State.WAIT
            self.waiting[passenger.origin].append(passenger)
            self.passengers.append(passenger)

    def get_buttonsIn(self):
        return np.array([bool(value) for value in self.onboarding.values()])
        #return np.array([len(value) if len(value)<=4 else 4 for value in self.onboarding.values()])
    
    def get_buttonsOut(self):
        return np.array([bool(value) for value in self.waiting.values()])
        #return np.array([len(value) if len(value)<=4 else 4 for value in self.waiting.values()])
    
    def waiting_passengers(self,tot_floor):
        num=0
        for floor in range(tot_floor):
            for passenger in self.waiting[floor]:
                num+=1
        return num

    def on_off_board(self,floor,T):
        tot_delayed_time=0.0
        for passenger in self.waiting[floor]:
            passenger.state = State.ONBOARD
            passenger.onboarding_time = T
            self.onboarding[passenger.dest].append(passenger)
        self.waiting[floor]=[]

        current_arrival = len(self.onboarding[floor])
        for passenger in self.onboarding[floor]:
            passenger.state = State.ARRIVAL
            tot_delayed_time+=(T-passenger.onboarding_time)-passenger.expected_arrival_time
            self.arrived[floor].append(passenger)
        self.onboarding[floor]=[]
        return current_arrival, tot_delayed_time
    
    def all_arrived(self):
        return all(passenger.state==State.ARRIVAL for passenger in self.passengers)
    
    def reset(self, passenger_args=None):
        if passenger_args == None:
            self.__init__(self.tot_floor, self.passenger_args)
        else:
            self.__init__(self.tot_floor, passenger_args)
        return

class ElevatorEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4, "passenger_modes":["determined","randomly_fixed","random_at_start","random_distribution"]}
    def __init__(self, render_mode="rgb_array",tot_floor=5, passenger_mode="random_distribution",passenger_num=5):
        
        ## set floor range
        self.tot_floor=tot_floor
        self.MIN_LOCATION = FLOOR_HEIGHT*(-EDGE_FLOOR_RANGE)
        self.MAX_LOCATION = FLOOR_HEIGHT*(self.tot_floor-1+EDGE_FLOOR_RANGE)
        self.ALLOWED_LOCATION = FLOOR_HEIGHT*(self.tot_floor-1)

        ## observation space & action space
        self.observation_space = spaces.Dict({
            "buttonsOut":spaces.MultiBinary(tot_floor),
            "buttonsIn":spaces.MultiBinary(tot_floor),
            "location":spaces.Box(low=self.MIN_LOCATION,high=self.MAX_LOCATION),
            "velocity":spaces.Box(low=MIN_VELOCITY,high=MAX_VELOCITY),
            "onFloor": spaces.Discrete(2)
            })
        self.action_space = spaces.Box(low=-10.0,high=10.0,dtype=np.float32)
        self.action=0
        self.T=0.0
        ## initialize passengerEnv
        assert passenger_mode in self.metadata["passenger_modes"]
        self.passenger_mode=passenger_mode
        if passenger_mode=="randomly_fixed" or passenger_mode=="random_at_start":
            self.passenger_num=passenger_num
            passenger_args=self.randomly_fix_passenger_args()
        elif passenger_mode=="random_distribution":
            passenger_args=self.random_distribution_passenger_args()
        self.passengerEnv=PassengerEnv(self.tot_floor,passenger_args)
              
        ## reward arguments
        self.cummulative_reward=0
        self.reward_args = {
            "accel_overload":0,
            "current_arrival":0,
            "non_stop":0,
            "tot_waiting_time":0,
            "tot_delayed_time":0
        }
        self.metric_args={
            "tot_passengers":0,
            "tot_waiting_time":0.0,
            "avg_delayed_time":0.0,
            "visited_floors":0,
            "rms_avg_actions":0.0
        }
        self.N=0

        
        self.start_state = dict({"buttonsOut":self.passengerEnv.get_buttonsOut(),
                                     "buttonsIn":self.passengerEnv.get_buttonsIn(),
                                     "location": np.array([np.random.randint(self.tot_floor)*FLOOR_HEIGHT], dtype=np.float32),
                                     "velocity": np.array([0.0], dtype=np.float32),
                                     "onFloor": None
                                     })
        self.start_state["onFloor"] = self.check_floor(self.start_state["location"], self.start_state["velocity"])[0]

        ## initialize rendering
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._init_render()
    
    def print_metric(self): 
        tot_passengers,tot_waiting_time,avg_delayed_time,visited_floors,rms_avg_actions=self.metric_args.values()
        print("tot_passengers: {}".format(tot_passengers))
        print("tot_waiting_time: {}".format(tot_waiting_time))
        print("avg_delayed_time: {}".format(avg_delayed_time))
        print("visited_floors: {}".format(visited_floors))
        print("rms_avg_actions: {}".format(rms_avg_actions))
        return tot_passengers,tot_waiting_time,avg_delayed_time,visited_floors,rms_avg_actions

    def randomly_fix_passenger_args(self):
        passenger_args=[]
        for i in range(self.passenger_num):
            passenger_origin=np.random.randint(0,self.tot_floor)
            passenger_dest=np.random.randint(0,self.tot_floor)
            while passenger_dest==passenger_origin:
                passenger_dest=np.random.randint(0,self.tot_floor)
            passenger_args.append((passenger_origin,passenger_dest,0.0))
        return passenger_args

    def random_distribution_passenger_args(self):
        passenger_distribution_factor = np.full(self.tot_floor, NORMAL_FLOOR_DISTRIBUTION_FACTOR)
        passenger_distribution_factor[0] = ZERO_FLOOR_DISTRIBUTION_FACTOR
        passenger_args=[]
        prob=np.random.rand(self.tot_floor)
        for origin in range(self.tot_floor):
            if prob[origin]<1-np.exp(-DELTA_T*passenger_distribution_factor[origin]):
                passenger_dest=np.random.randint(0,self.tot_floor)
                while passenger_dest==origin:
                    passenger_dest=np.random.randint(0,self.tot_floor)
                passenger_args.append((origin,passenger_dest,self.T))
        return passenger_args

    def check_floor(self, location, velocity):
        location = location[0] / FLOOR_HEIGHT
        round_location = round(location)
        velocity = velocity[0]
        if 0<=round_location and round_location<self.tot_floor and abs(round_location-location)<FLOOR_RANGE:
            on_floor = 1
            self.metric_args["visited_floors"]+=1
            if abs(velocity)<STOP_VEL_RANGE:
                return on_floor, True, round(location)
        else:
            on_floor = 0
            if abs(velocity)<(STOP_VEL_RANGE/2):
                self.reward_args["non_stop"] = (STOP_VEL_RANGE/2) - abs(velocity)
        return on_floor, False, None 
        
    def on_off_board(self, floor):
        current_arrival , tot_delayed_time= self.passengerEnv.on_off_board(floor,self.T)
        self.reward_args["current_arrival"] = current_arrival
        self.reward_args["tot_delayed_time"]=tot_delayed_time
        if current_arrival!=0:
            self.metric_args["avg_delayed_time"]=(self.metric_args["avg_delayed_time"]*self.metric_args["tot_passengers"]
                                                  +tot_delayed_time)/(self.metric_args["tot_passengers"]+current_arrival)
            self.metric_args["tot_passengers"]+=current_arrival
                                                
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
        self.reward_args["tot_waiting_time"]=self.passengerEnv.waiting_passengers(self.tot_floor)*DELTA_T
        self.metric_args["tot_waiting_time"]+=self.reward_args["tot_waiting_time"]
        accel_overLoad, current_arrival, non_stop, tot_waiting_time,tot_delayed_time= self.reward_args.values()
        reward = (current_arrival)*CURRIVAL_REWARD+(non_stop)*NONSTOP_REWARD+(tot_delayed_time)*DELAYED_TIME_REWARD+tot_waiting_time*WAITING_TIME_REWARD
        self.reward_args = {key: 0 for key in self.reward_args}
        return reward
    
    def reset(self, seed=None, options=None):
        self.cummulative_reward=0
        self.T=0.0
        self.action=0
        self.N=0
        self.metric_args = {key: 0 for key in self.metric_args}
        if self.passenger_mode=="random_at_start":
            passenger_args=self.randomly_fix_passenger_args()
            self.passengerEnv.reset(passenger_args)
        else: 
            self.passengerEnv.reset()
        
        self.observation = dict({"buttonsOut":self.passengerEnv.get_buttonsOut(),
                                "buttonsIn":self.passengerEnv.get_buttonsIn(),
                                "location": np.array([0.0], dtype=np.float32),
                                "velocity": np.array([0.0], dtype=np.float32),
                                "onFloor": None
                                })
        self.observation["onFloor"] = self.check_floor(self.start_state["location"], self.start_state["velocity"])[0]
        return self.observation,{"info":None}
        
    def step(self, action):
        self.N+=1
        self.action=action[0]
        self.metric_args["rms_avg_actions"]=np.sqrt((np.square(self.metric_args["rms_avg_actions"])*(self.N-1)+np.square(action[0]))/self.N)
        prev_state=self.observation
        next_state=copy.deepcopy(prev_state)
        
        next_state["location"]+=DELTA_T*next_state['velocity']+0.5*action*pow(DELTA_T,2)
        next_state['velocity']+=DELTA_T*action
        self.clip(next_state)

        if self.passenger_mode=="random_distribution":
            passenger_args=self.random_distribution_passenger_args()
            self.passengerEnv.create(passenger_args)
            next_state['buttonsOut']=self.passengerEnv.get_buttonsOut()
            next_state['buttonsIn']=self.passengerEnv.get_buttonsIn()
        
        on_floor, stop_on_floor, floor=self.check_floor(next_state["location"], next_state["velocity"])
        next_state["onFloor"] = on_floor
        if stop_on_floor:
            self.on_off_board(floor)
            next_state['buttonsOut']=self.passengerEnv.get_buttonsOut()
            next_state['buttonsIn']=self.passengerEnv.get_buttonsIn()
        self.observation=next_state        
        self.T+=DELTA_T
        reward=self.compute_reward(prev_state,action,next_state)
        self.cummulative_reward+=reward
        
        if self.passenger_mode=="random_distribution":
            done = False
        else:
            done = self.passengerEnv.all_arrived()
        tot_passengers,tot_waiting_time,avg_delayed_time,visited_floors,rms_avg_actions=self.metric_args.values()
        return self.observation,reward,done,False,{"tot_passengers":tot_passengers,"tot_waiting_time":tot_waiting_time,
                                                   "avg_delayed_time":avg_delayed_time,"visited_floors":visited_floors,
                                                   "rms_avg_actions":rms_avg_actions}
    
    def _init_render(self):
        pygame.font.init()
        self.font=pygame.font.Font('freesansbold.ttf',30)
        #self.numfont=pygame.font.Font('freesansbold.ttf',30)

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
        return

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        black=(0,0,0)
        gray=(224,224,224)
        white=(255,255,255)
        color=[(255,184,0),(255,0,138),(82,0,255),(0,240,255),(173,255,0)]
        buttons=[white,black]

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(white)
        pix_square_size = (
            self.window_size / self.tot_floor
        )  # The size of a single grid square in pixels
        elevator_height=pix_square_size/3
        elevator_width=elevator_height/2

        floor_pixel_size=(self.window_size-pix_square_size)/(self.tot_floor-1)
        borderLine_thickness=floor_pixel_size*FLOOR_RANGE

        # First we draw the target
        for i in range(self.tot_floor):
            if i!=self.tot_floor-1:
                pygame.draw.line(
                    canvas,
                    gray,
                    (self.window_size/2,(i+0.5)*pix_square_size),
                    (self.window_size/2,(i+1.5)*pix_square_size),
                    width=10,
                )
            pygame.draw.rect(
                canvas,
                color[self.tot_floor-i-1],
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
            
            pygame.draw.rect(
                canvas,
                white,
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
                buttons[self.observation['buttonsIn'][i]],
                (self.window_size -60,self.window_size-(i+0.5)*pix_square_size),
                20,
            )
            pygame.draw.circle(
                canvas,
                buttons[self.observation['buttonsOut'][i]],
                (60,self.window_size-(i+0.5)*pix_square_size),
                20,
            )

        pygame.draw.rect(
            canvas,
            black,
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
                    color[self.passengerEnv.passengers[i].dest],
                    (self.window_size/2 +80 + 40*min(6, num_arrived[self.passengerEnv.passengers[i].dest]) ,self.window_size-(self.passengerEnv.passengers[i].dest+0.5)*pix_square_size),
                    15,
                )
                # origin_text=self.numfont.render(str(self.passengerEnv.passengers[i].origin) ,True,(255,255,255))
                # origin_text_rect=origin_text.get_rect()
                # origin_text_rect.center=(self.window_size/2 +100 + 50*num_arrived[self.passengerEnv.passengers[i].dest],self.window_size-(self.passengerEnv.passengers[i].dest+0.5)*pix_square_size )
                # canvas.blit(origin_text,origin_text_rect)
                num_arrived[self.passengerEnv.passengers[i].dest]+=1
            elif self.passengerEnv.passengers[i].state is State.WAIT:
                pygame.draw.circle(
                    canvas,
                    color[self.passengerEnv.passengers[i].dest],
                    (self.window_size/2 -80- 40*min(6, num_waiting[self.passengerEnv.passengers[i].origin]) ,self.window_size-(self.passengerEnv.passengers[i].origin+0.5)*pix_square_size ),
                    15,
                )
                # dest_text=self.numfont.render(str(self.passengerEnv.passengers[i].dest),True,(255,255,255))
                # dest_text_rect=dest_text.get_rect()
                # dest_text_rect.center=(self.window_size/2 -100 -50*num_waiting[self.passengerEnv.passengers[i].origin] ,self.window_size-(self.passengerEnv.passengers[i].origin+0.5)*pix_square_size )
                # canvas.blit(dest_text,dest_text_rect)
                num_waiting[self.passengerEnv.passengers[i].origin]+=1

        # loc_text=self.font.render("loc(m): "+str(round(self.observation['location'][0],2)),True,(0,0,0))
        # loc_text_rect=loc_text.get_rect()
        # loc_text_rect.center=(150,40)
        # canvas.blit(loc_text,loc_text_rect)

        # vel_text=self.font.render("vel(m/s) : "+str(round(self.observation['velocity'][0],2)),True,(0,0,0))
        # vel_text_rect=vel_text.get_rect()
        # vel_text_rect.center=(150,100)
        # canvas.blit(vel_text,vel_text_rect)

        action_text=self.font.render("action : "+str(round(self.action,2)),True,(0,0,0))
        action_text_rect=action_text.get_rect()
        action_text_rect.center=(150,30)
        canvas.blit(action_text,action_text_rect)

        rew_text=self.font.render("rew : "+str(round(self.cummulative_reward,2)),True,(0,0,0))
        rew_text_rect=rew_text.get_rect()
        rew_text_rect.center=(800,30)
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

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

class ButtonElevatorEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4, "passenger_modes":["determined","randomly_fixed","random_at_start","random_distribution"]}
    def __init__(self, render_mode="rgb_array",tot_floor=5, passenger_mode="random_distribution",passenger_num=5):
        
        ## set floor range
        self.tot_floor=tot_floor
        self.MIN_LOCATION = FLOOR_HEIGHT*(-EDGE_FLOOR_RANGE)
        self.MAX_LOCATION = FLOOR_HEIGHT*(self.tot_floor-1+EDGE_FLOOR_RANGE)
        self.ALLOWED_LOCATION = FLOOR_HEIGHT*(self.tot_floor-1)

        ## observation space & action space
        self.observation_space = spaces.Dict({
            "buttonsOut":spaces.MultiDiscrete([5]*tot_floor),
            "buttonsIn":spaces.MultiDiscrete([5]*tot_floor),
            "location":spaces.Box(low=self.MIN_LOCATION,high=self.MAX_LOCATION),
            "velocity":spaces.Box(low=MIN_VELOCITY,high=MAX_VELOCITY),
            "onFloor": spaces.Discrete(2)
            })
        self.action_space = spaces.Box(low=-10.0,high=10.0,dtype=np.float32)
        self.action=0
        self.T=0.0
        ## initialize passengerEnv
        assert passenger_mode in self.metadata["passenger_modes"]
        self.passenger_mode=passenger_mode
        if passenger_mode=="randomly_fixed" or passenger_mode=="random_at_start":
            self.passenger_num=passenger_num
            passenger_args=self.randomly_fix_passenger_args()
        elif passenger_mode=="random_distribution":
            passenger_args=self.random_distribution_passenger_args()
        self.passengerEnv=PassengerEnv(self.tot_floor,passenger_args)
              
        ## reward arguments
        self.cummulative_reward=0
        self.reward_args = {
            "accel_overload":0,
            "current_arrival":0,
            "non_stop":0,
            "tot_waiting_time":0,
            "tot_delayed_time":0
        }
        self.metric_args={
            "tot_passengers":0,
            "tot_waiting_time":0.0,
            "avg_delayed_time":0.0,
            "visited_floors":0,
            "rms_avg_actions":0.0
        }
        self.N=0

        
        self.start_state = dict({"buttonsOut":self.passengerEnv.get_buttonsOut(),
                                     "buttonsIn":self.passengerEnv.get_buttonsIn(),
                                     "location": np.array([np.random.randint(self.tot_floor)*FLOOR_HEIGHT], dtype=np.float32),
                                     "velocity": np.array([0.0], dtype=np.float32),
                                     "onFloor": None
                                     })
        self.start_state["onFloor"] = self.check_floor(self.start_state["location"], self.start_state["velocity"])[0]

        ## initialize rendering
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._init_render()
    
    def print_metric(self): 
        tot_passengers,tot_waiting_time,avg_delayed_time,visited_floors,rms_avg_actions=self.metric_args.values()
        print("tot_passengers: {}".format(tot_passengers))
        print("tot_waiting_time: {}".format(tot_waiting_time))
        print("avg_delayed_time: {}".format(avg_delayed_time))
        print("visited_floors: {}".format(visited_floors))
        print("rms_avg_actions: {}".format(rms_avg_actions))
        return tot_passengers,tot_waiting_time,avg_delayed_time,visited_floors,rms_avg_actions

    def randomly_fix_passenger_args(self):
        passenger_args=[]
        for i in range(self.passenger_num):
            passenger_origin=np.random.randint(0,self.tot_floor)
            passenger_dest=np.random.randint(0,self.tot_floor)
            while passenger_dest==passenger_origin:
                passenger_dest=np.random.randint(0,self.tot_floor)
            passenger_args.append((passenger_origin,passenger_dest,0.0))
        return passenger_args

    def random_distribution_passenger_args(self):
        passenger_distribution_factor = np.full(self.tot_floor, NORMAL_FLOOR_DISTRIBUTION_FACTOR)
        passenger_distribution_factor[0] = ZERO_FLOOR_DISTRIBUTION_FACTOR
        passenger_args=[]
        prob=np.random.rand(self.tot_floor)
        for origin in range(self.tot_floor):
            if prob[origin]<1-np.exp(-DELTA_T*passenger_distribution_factor[origin]):
                passenger_dest=np.random.randint(0,self.tot_floor)
                while passenger_dest==origin:
                    passenger_dest=np.random.randint(0,self.tot_floor)
                passenger_args.append((origin,passenger_dest,self.T))
        return passenger_args

    def check_floor(self, location, velocity):
        location = location[0] / FLOOR_HEIGHT
        round_location = round(location)
        velocity = velocity[0]
        if 0<=round_location and round_location<self.tot_floor and abs(round_location-location)<FLOOR_RANGE:
            on_floor = 1
            self.metric_args["visited_floors"]+=1
            if abs(velocity)<STOP_VEL_RANGE:
                return on_floor, True, round(location)
        else:
            on_floor = 0
            if abs(velocity)<(STOP_VEL_RANGE/2):
                self.reward_args["non_stop"] = (STOP_VEL_RANGE/2) - abs(velocity)
        return on_floor, False, None 
        
    def on_off_board(self, floor):
        current_arrival , tot_delayed_time= self.passengerEnv.on_off_board(floor,self.T)
        self.reward_args["current_arrival"] = current_arrival
        self.reward_args["tot_delayed_time"]=tot_delayed_time
        if current_arrival!=0:
            self.metric_args["avg_delayed_time"]=(self.metric_args["avg_delayed_time"]*self.metric_args["tot_passengers"]
                                                  +tot_delayed_time)/(self.metric_args["tot_passengers"]+current_arrival)
            self.metric_args["tot_passengers"]+=current_arrival
                                                
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
        self.reward_args["tot_waiting_time"]=self.passengerEnv.waiting_passengers(self.tot_floor)*DELTA_T
        self.metric_args["tot_waiting_time"]+=self.reward_args["tot_waiting_time"]
        accel_overLoad, current_arrival, non_stop, tot_waiting_time,tot_delayed_time= self.reward_args.values()
        reward = (current_arrival)*CURRIVAL_REWARD+(non_stop)*NONSTOP_REWARD+(tot_delayed_time)*DELAYED_TIME_REWARD+tot_waiting_time*WAITING_TIME_REWARD
        self.reward_args = {key: 0 for key in self.reward_args}
        return reward
    
    def reset(self, seed=None, options=None):
        self.cummulative_reward=0
        self.T=0.0
        self.action=0
        self.N=0
        self.metric_args = {key: 0 for key in self.metric_args}
        if self.passenger_mode=="random_at_start":
            passenger_args=self.randomly_fix_passenger_args()
            self.passengerEnv.reset(passenger_args)
        else: 
            self.passengerEnv.reset()
        
        self.observation = dict({"buttonsOut":self.passengerEnv.get_buttonsOut(),
                                "buttonsIn":self.passengerEnv.get_buttonsIn(),
                                "location": np.array([0.0], dtype=np.float32),
                                "velocity": np.array([0.0], dtype=np.float32),
                                "onFloor": None
                                })
        self.observation["onFloor"] = self.check_floor(self.start_state["location"], self.start_state["velocity"])[0]
        return self.observation,{"info":None}
        
    def step(self, action):
        self.N+=1
        self.action=action[0]
        self.metric_args["rms_avg_actions"]=np.sqrt((np.square(self.metric_args["rms_avg_actions"])*(self.N-1)+np.square(action[0]))/self.N)
        prev_state=self.observation
        next_state=copy.deepcopy(prev_state)
        
        next_state["location"]+=DELTA_T*next_state['velocity']+0.5*action*pow(DELTA_T,2)
        next_state['velocity']+=DELTA_T*action
        self.clip(next_state)

        if self.passenger_mode=="random_distribution":
            passenger_args=self.random_distribution_passenger_args()
            self.passengerEnv.create(passenger_args)
            next_state['buttonsOut']=self.passengerEnv.get_buttonsOut()
            next_state['buttonsIn']=self.passengerEnv.get_buttonsIn()
        
        on_floor, stop_on_floor, floor=self.check_floor(next_state["location"], next_state["velocity"])
        next_state["onFloor"] = on_floor
        if stop_on_floor:
            self.on_off_board(floor)
            next_state['buttonsOut']=self.passengerEnv.get_buttonsOut()
            next_state['buttonsIn']=self.passengerEnv.get_buttonsIn()
        self.observation=next_state        
        self.T+=DELTA_T
        reward=self.compute_reward(prev_state,action,next_state)
        self.cummulative_reward+=reward
        
        if self.passenger_mode=="random_distribution":
            done = False
        else:
            done = self.passengerEnv.all_arrived()
        tot_passengers,tot_waiting_time,avg_delayed_time,visited_floors,rms_avg_actions=self.metric_args.values()
        return self.observation,reward,done,False,{"tot_passengers":tot_passengers,"tot_waiting_time":tot_waiting_time,
                                                   "avg_delayed_time":avg_delayed_time,"visited_floors":visited_floors,
                                                   "rms_avg_actions":rms_avg_actions}
    
    def _init_render(self):
        pygame.font.init()
        self.font=pygame.font.Font('freesansbold.ttf',30)
        #self.numfont=pygame.font.Font('freesansbold.ttf',30)

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
        return

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        black=(0,0,0)
        gray=(224,224,224)
        white=(255,255,255)
        color=[(255,184,0),(255,0,138),(82,0,255),(0,240,255),(173,255,0)]
        buttons=[black,(0,255,0)]

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(white)
        pix_square_size = (
            self.window_size / self.tot_floor
        )  # The size of a single grid square in pixels
        elevator_height=pix_square_size/3
        elevator_width=elevator_height/2

        floor_pixel_size=(self.window_size-pix_square_size)/(self.tot_floor-1)
        borderLine_thickness=floor_pixel_size*FLOOR_RANGE

        # First we draw the target
        for i in range(self.tot_floor):
            pygame.draw.rect(
                canvas,
                color[i],
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
                    gray,
                    (self.window_size/2,(i+0.5)*pix_square_size),
                    (self.window_size/2,(i+1.5)*pix_square_size),
                    width=10,
                )
            pygame.draw.rect(
                canvas,
                white,
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
                buttons[bool(self.observation['buttonsIn'][i])],
                (self.window_size -100,self.window_size-(i+0.5)*pix_square_size),
                20,
            )
            pygame.draw.circle(
                canvas,
                buttons[bool(self.observation['buttonsOut'][i])],
                (100,self.window_size-(i+0.5)*pix_square_size),
                20,
            )

        pygame.draw.rect(
            canvas,
            black,
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
                    color[self.passengerEnv.passengers[i].dest],
                    (self.window_size/2 +100 + 50*num_arrived[self.passengerEnv.passengers[i].dest] ,self.window_size-(self.passengerEnv.passengers[i].dest+0.5)*pix_square_size),
                    20,
                )
                # origin_text=self.numfont.render(str(self.passengerEnv.passengers[i].origin) ,True,(255,255,255))
                # origin_text_rect=origin_text.get_rect()
                # origin_text_rect.center=(self.window_size/2 +100 + 50*num_arrived[self.passengerEnv.passengers[i].dest],self.window_size-(self.passengerEnv.passengers[i].dest+0.5)*pix_square_size )
                # canvas.blit(origin_text,origin_text_rect)
                num_arrived[self.passengerEnv.passengers[i].dest]+=1
            elif self.passengerEnv.passengers[i].state is State.WAIT:
                pygame.draw.circle(
                    canvas,
                    color[self.passengerEnv.passengers[i].dest],
                    (self.window_size/2 -100- 50*num_waiting[self.passengerEnv.passengers[i].origin] ,self.window_size-(self.passengerEnv.passengers[i].origin+0.5)*pix_square_size ),
                    20,
                )
                # dest_text=self.numfont.render(str(self.passengerEnv.passengers[i].dest),True,(255,255,255))
                # dest_text_rect=dest_text.get_rect()
                # dest_text_rect.center=(self.window_size/2 -100 -50*num_waiting[self.passengerEnv.passengers[i].origin] ,self.window_size-(self.passengerEnv.passengers[i].origin+0.5)*pix_square_size )
                # canvas.blit(dest_text,dest_text_rect)
                num_waiting[self.passengerEnv.passengers[i].origin]+=1

        # loc_text=self.font.render("loc(m): "+str(round(self.observation['location'][0],2)),True,(0,0,0))
        # loc_text_rect=loc_text.get_rect()
        # loc_text_rect.center=(150,40)
        # canvas.blit(loc_text,loc_text_rect)

        # vel_text=self.font.render("vel(m/s) : "+str(round(self.observation['velocity'][0],2)),True,(0,0,0))
        # vel_text_rect=vel_text.get_rect()
        # vel_text_rect.center=(150,100)
        # canvas.blit(vel_text,vel_text_rect)

        action_text=self.font.render("action : "+str(round(self.action,2)),True,(0,0,0))
        action_text_rect=action_text.get_rect()
        action_text_rect.center=(150,30)
        canvas.blit(action_text,action_text_rect)

        rew_text=self.font.render("rew : "+str(round(self.cummulative_reward,2)),True,(0,0,0))
        rew_text_rect=rew_text.get_rect()
        rew_text_rect.center=(800,30)
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

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()