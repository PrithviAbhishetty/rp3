#import math packages
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random
import os

#import gym packages
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

#import stable baselines packages
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

show_animation = True

ox, oy = [], []

def generate_gaussian_grid_map(uxp, uyp, uxn, uyn, xyreso, std):
    minx = 0
    miny = 0
    maxx = 51
    maxy = 51
    neg = False
    xw = int(round((maxx - minx) / xyreso))
    yw = int(round((maxy - miny) / xyreso))

    #create a gaussian filter with zero intensity
    gmap = [[0.0 for i in range(yw)] for i in range(xw)]

    #loop through all cells
    for ix in range(xw):
        for iy in range(yw):

            x = ix * xyreso + minx
            y = iy * xyreso + miny

            mindis = float("inf")
            #find the minimum distance to closest Orographic source
            for (iuxp, iuyp) in zip(uxp, uyp):
                dp = math.hypot(iuxp - x, iuyp - y)
                if mindis >= dp:
                    mindis = dp

            for (iuxn, iuyn) in zip(uxn, uyn):
                dn = math.hypot(iuxn - x, iuyn - y)
                if mindis >= dn:
                    mindis = dn
                    neg = True

            #create probability density function
            pdf = (1.0 - norm.cdf(mindis, 0.0, std))
            #update intensity in gaussian map
            if neg == True:
                gmap[ix][iy] = -pdf
            else:
                gmap[ix][iy] = pdf

    #set gaussian map intensity to zero at all obstacle cells
    for (a,b) in zip(ox,oy):
        gmap[int(a)][int(b)] = 0.0
    return gmap

def draw_heatmap(data, minx, maxx, miny, maxy, xyreso):
    data = np.absolute(data)
    x, y = np.mgrid[slice(minx - xyreso / 2.0, maxx + xyreso / 2.0, xyreso),
                    slice(miny - xyreso / 2.0, maxy + xyreso / 2.0, xyreso)]
    plt.pcolor(x, y, data, vmax=1.0, cmap=plt.cm.Blues)
    plt.axis("equal")

def square(x1, x2, y1, y2):
    for i in range(x1, x2+1):
        for j in range(y1, y2+1):
            if i != 0 and i != 50 and j != 0 and j != 50:
                ox.append(i)
                oy.append(j)

def Updraft():
    uxp, uyp, uxn, uyn = [], [], [], []
    i = 0
    j = 0
    match = False
    while i+j < 8:
        #print(i)
        match = False
        sign = random.randint(0,1)
        if sign == 0:
            uxp.append(np.random.randint(1, 49, 1))
            uyp.append(np.random.randint(1, 49, 1))
            for (a,b) in zip(ox,oy):
                if uxp[i] == a and uyp[i] == b:
                    match = True
            if match == True:
                uxp.pop()
                uyp.pop()
                i = i - 1
            i = i + 1
        if sign == 1:
            uxn.append(np.random.randint(1, 49, 1))
            uyn.append(np.random.randint(1, 49, 1))
            for (a,b) in zip(ox,oy):
                if uxn[j] == a and uyn[j] == b:
                    match = True
            if match == True:
                uxn.pop()
                uyn.pop()
                j = j - 1
            j = j + 1
    return uxp, uyp, uxn, uyn

def GridMap():

    # start and goal position
    sx = 48  # [m]
    sy = 20  # [m]
    gx = [8]  # [m]
    gy = [44]  # [m]

    #obstacle building positions
    square(10, 15, 10, 15)
    square(10, 14, 30, 35)
    square(40, 45, 20, 25)
    square(30, 35, 40 , 45)
    square(24, 30, 0, 9)
    square(18, 21, 33, 45)
    square(44, 50, 3, 6)
    square(27, 30, 24 , 29)

    #obstacle boundary
    for i in range(0, 51):
        ox.append(i)
        oy.append(0)
        ox.append(i)
        oy.append(50)
    for i in range(1, 50):
        ox.append(0)
        oy.append(i)
        ox.append(50)
        oy.append(i)

    xyreso = 1.0  # xy grid resolution
    STD = 5.0  # standard diviation for gaussian distribution

    #random generation of orographic updrafts and downdrafts
    uxp, uyp, uxn, uyn = Updraft()

    #getting gaussian grid data
    gmap = generate_gaussian_grid_map(
        uxp, uyp, uxn, uyn, xyreso, STD)

    #plotting
    if show_animation:
        #plot obstacles
        plt.plot(ox, oy, ".k")
        #plot start point
        plt.plot(sx, sy, "og")
        #plot goal point
        plt.plot(gx, gy, "xb")
        plt.axis("equal")

    if show_animation:
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        #render of gaussian distribution
        draw_heatmap(gmap, minx=0, maxx=51, miny=0, maxy=51, xyreso=1.0)
        #plot updrafts
        plt.plot(uxp, uyp, ".r")
        #plot downdrafts
        plt.plot(uxn, uyn, ".m")
        plt.grid(True)
        plt.legend(['Obstacle', 'Start','Goal','Updraft','Downdraft',
        'Wind-Field'],bbox_to_anchor=(0.86, 1), loc='upper left')

    return sx, sy, gx, gy, ox, oy, uxp, uyp, uxn, uyn, gmap

#run instance of grid map, returning the start, goal, obstacle, updraft, downdraft postions and gaussian distribution data.
sx, sy, gx, gy, ox, oy, uxp, uyp, uxn, uyn, gmap = GridMap()

#Creating the Reinforcement Learning Environment in the Markov Decision Process
class GridEnv(Env):
    #initialise class
    def __init__(self):
        self.action_space = Discrete(4)
        self.observation_space = MultiDiscrete([50,50])
        self.state = np.array([sx,sy])
        self.planning_length = 2500
        self.visit_x = [sx]
        self.visit_y = [sy]
        self.uxp, self.uyp, self.uxn, self.uyn = Updraft()
        self.gmap = generate_gaussian_grid_map(self.uxp, self.uyp, self.uxn, self.uyn, 1.0, 5.0)
        self.counter = 0
        self.subscore = 0
        self.cost = 0

    def step(self, action):
        # update state / grid position by matching selected action to a cartesian transformation
        if action == 0:
            self.state += 1,0
        elif action == 1:
            self.state += -1,0
        elif action == 2:
            self.state += 0,1
        elif action == 3:
            self.state += 0,-1

        repeat = False
        reward = 0

        #Visit reward
        for a,b in zip(self.visit_x, self.visit_y):
            if (self.state[0] == a) and (self.state[1] == b):
                repeat = True

        if repeat == True:
            visitreward = -1
        else:
            visitreward = 1
            self.counter += 1

        self.visit_x.append(self.state[0])
        self.visit_y.append(self.state[1])

        # decrement planning length and calculate Time reward
        self.planning_length -= 1
        timereward = -0.002*int(2500-self.planning_length)

        end = False
        #Obstacle reward
        for (a,b) in zip(ox,oy):
            if self.state[0] == a and self.state[1] == b:
                reward = 0
                end = True
        if end == False:
            #Orographic reward
            reward = 0.6*(self.gmap[int(self.state[0])][int(self.state[1])])
            for (c,d) in zip(gx,gy):
                if self.state[0] == c and self.state[1] == d:
                    #Goal reward
                    reward = 1000
                    end = True
                    print('goal reached')

        #single returned reward value
        reward += (timereward + visitreward)
        self.subscore += reward

        #Check if agent is at a terminal state
        if self.planning_length <= 0 or end == True:
            done = True
            for n in range(2500-self.planning_length):
                #calculating flight cost
                self.cost += 2-(2*gmap[self.visit_x[n]][self.visit_y[n]])
                #self.cost += 2
            self.cost = int(self.cost)
            self.subscore = int(self.subscore)
            print('ended at {}, length: {}, new positions: {}, ep score: {}, flight cost: {}'.format(self.state, 2500-self.planning_length, self.counter, self.subscore, self.cost))
        else:
            done = False

        info = {}

        return self.state, reward, done, info

    def render(self, mode ='human'):
        #plotting
        if mode=='human':
            plt.clf()
            plt.plot(ox, oy, ".k")
            plt.plot(sx, sy, "og")
            plt.plot(gx, gy, "xb")
            plt.axis("equal")
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            draw_heatmap(self.gmap, 0, 51, 0, 51, 1)
            plt.plot(self.uxp, self.uyp, ".r")
            plt.plot(self.uxn, self.uyn, ".m")
            plt.title("2D Grid Wind-Field Environment")
            plt.xlabel('Terminal State {}, Length: {}, Reward: {}, Flight Cost: {}'.format(self.state, 2500-self.planning_length, self.subscore, self.cost))
            plt.grid(True)
            #plotting path taken by agent in a single episode
            plt.plot(self.visit_x, self.visit_y, "-g")
            plt.pause(0.0001)

    #reset environment for next episode
    def reset(self):
        self.state = np.array([sx,sy])
        self.planning_length = 2500
        self.visit_x = [sx]
        self.visit_y = [sy]
        self.uxp, self.uyp, self.uxn, self.uyn = Updraft()
        self.gmap = generate_gaussian_grid_map(self.uxp, self.uyp, self.uxn, self.uyn, 1.0, 5.0)
        self.counter = 0
        self.subscore = 0
        self.cost = 0
        return self.state

#call the environment class
env = GridEnv()

env.reset()

#test the path planning agent for 10 episodes, pre-training
episodes = 10
for episodes in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    while not done:
        #take random actions
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episodes, score))
    env.render()
env.close()

#create a storage path for training logs
log_path = os.path.join('Training', 'Logs')

#create a new instance of the algorithm by wrapping the environment with an MLP policy and PPO
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

#train the model for a number of time-steps
model.learn(total_timesteps=100000)

#create a file to store the trained path planner
GridEnv_path = os.path.join('Training', 'Saved Models','Main_3.5_PPO')

#load trained planner at the file location
model = PPO.load(GridEnv_path, env)

#save the planner at the file location
model.save(GridEnv_path)

#delete the model
del model

#re-load the model
model = PPO.load(GridEnv_path, env)

#evaluate the learned policy with a deterministic approach
datapoint1, datapoint2 = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True, render=False, callback=None, reward_threshold=None, return_episode_rewards=True)

#output reward and length of path determined optimum
print(datapoint1)
print("--")
print(datapoint2)

#test the path planning agent for 10 episodes, post-training
episodes = 10
for episodes in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        #algorithm predicts action using trained policy, with a stochastic approach
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episodes, score))
    env.render()
    plt.show()
env.close()
