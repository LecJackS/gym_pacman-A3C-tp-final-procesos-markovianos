"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

#import gym_super_mario_bros
import gym
import gym_pacman

from gym.spaces import Box
from gym import Wrapper
#from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from nes_py.wrappers import JoypadSpace
#from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import numpy as np
import subprocess as sp


class Monitor:
    def __init__(self, width, height, saved_path):
        # conda install ffmpeg
        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "80", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())
        #print(len(image_array))

def process_frame(frame):
    if frame is not None:
        # frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # resize to 84x84px; normalize color num to [0,1)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))



class SameReward(Wrapper):
    def __init__(self, env=None, monitor=None):
        super(SameReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        # Process frame to 84x84px grayscale 
        state = process_frame(state)
        #print("info:", info)
        reward = info["score"] / 1000.

        #print("\naction:",action,"  reward:", reward)
        #print(info, "\n")
        return state, reward, done, info

    def reset(self):
        return process_frame(self.env.reset())
    
class DifReward(Wrapper):
    def __init__(self, env=None, monitor=None):
        super(DifReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        # Process frame to 84x84px grayscale 
        state = process_frame(state)
        #print("info:", info)
        reward += (info["score"] - self.curr_score) / 10.
        #print("reward parcial:",reward)
        self.curr_score = info["score"]
        # for pacman gym
        #if done:
        #    if info["max_ep"]:
        #        # max episode lenght reached
        #        reward -= 500
        #        print("penalty! info:", info["max_ep"])
        return state, reward, done, info

    def reset(self):
        self.curr_score = 0
        return process_frame(self.env.reset())

class CustomReward(Wrapper):
    def __init__(self, env=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        
        if self.monitor:
            self.monitor.record(state)
        # Process frame to 84x84px grayscale 
        state = process_frame(state)
        #print("info:", info)
        reward += (info["score"] - self.curr_score) / 10.
        #print("reward:",reward)
        self.curr_score = info["score"]
        # for mario
        #if done:
        #    if info["flag_get"]:
        #        reward += 50
        #    else:
        #        reward -= 50
        # for pacman gym
        #if done:
        #    if info["max_ep"]:
        #        # max episode lenght reached
        #        reward -= 500
        #        print("penalty! info:", info["max_ep"])
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        return process_frame(self.env.reset())
    
class NoSkipFrame(Wrapper):
    """ Neural network four frame input:
        [T,
         T,
         T,
         T] """
    def __init__(self, env, skip=4):
        super(NoSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.skip = skip

    def step(self, action):
        states = []
        #done = self.done
        state, reward, done, info = self.env.step(action)
        # we need a four channel input
        copies = 4 # instead of copies, do transformations! <<<  TODO
        for i in range(copies):
            states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(np.float32), reward, done, info
    
    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        return states.astype(np.float32)

class NoSkipFrameFourRotations(Wrapper):
    """ Neural network four frame input:
        [T,
         rot90(T),
         rot180(T),
         rot270(T)] """
    def __init__(self, env, skip=4):
        super(NoSkipFrameFourRotations, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.skip = skip

    def step(self, action):
        states = []
        #done = self.done
        state, reward, done, info = self.env.step(action)
        # we need a four channel input 
        # We'll get all four rotations of current fram
        # TODO: mirroring horizontal and diagonal (4 more)
        states.append(state)
        times = 3 # instead of copies, do transformations! <<<  TODO/ DONE! :)
        for t in range(1, times+1):
            # rotate +90 degrees each frame 
            states.append(np.rot90(state, t, axes=(1, 2)))
        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(np.float32), reward, done, info
    
    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        return states.astype(np.float32)
    
class MinimSkipFrame(Wrapper):
    """ Neural network four frame input:
        [T,
         T,
         T+1,
         T+1] """
    def __init__(self, env, skip=4):
        super(MinimSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.skip = skip

    def step(self, action):
        total_reward = 0
        states = []
        #done = self.done
        state, reward, done, info = self.env.step(action)
        states.append(state)
        states.append(state) # we need a four channel input
        total_reward += reward
        # we pass states in mini groups of 2
        if not done:
            state, reward, done, info = self.env.step(action)
            states.append(state)
            states.append(state)
            total_reward += reward
        else:
            states.append(state)
            states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        print(states)
        print(states.astype(np.float32))
        return states.astype(np.float32), reward, done, info
    
    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        return states.astype(np.float32)

class SimpleSkipFrame(Wrapper):
    """ Neural network four frame input:
        [T,
         T+4,
         T+8,
         T+12] """
    def __init__(self, env, skip=4):
        super(SimpleSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.skip = skip

    def step(self, action):
        total_reward = 0
        states = []
        #done = self.done
        state, reward, done, info = self.env.step(action)
        total_reward += reward
        # we pass states in mini groups of skip=4
        for i in range(self.skip):
            if not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                states.append(state)
            else:
                states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(np.float32), reward, done, info

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        return states.astype(np.float32)
    
class DQNSkipFrame(Wrapper):
    """ # https://github.com/openai/gym/issues/275
        # (tried to be) implemented by jack 
        Neural network four frame input:
        [max(T-1,  T),
         max(T+3,  T+4),
         max(T+7,  T+8),
         max(T+11, T+12)] """
    def __init__(self, env, skip=4):
        super(DQNSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.skip = skip

    def step(self, action):
        total_reward = 0
        states = []

        # we pass states in mini groups of skip=4
        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if not done:
                state2, reward, done, info = self.env.step(action)
                total_reward += reward
                # element wise max
                state = np.maximum(state, state2)
            states.append(state)
            for j in range(2):
                #dummy actions, but we keep track of reward
                _,reward,done,_ = self.env.step(action)
                total_reward += reward
#                 if done:
#                     break #this dummy loop
        total_reward /= 12.
        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(np.float32), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        return states.astype(np.float32)

class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.skip = skip

    def step(self, action):
        total_reward = 0
        states = []
        # skips one frame, and pass the next four to the NN
        state, reward, done, info = self.env.step(action)
        # we pass states in mini groups of skip=4
        for i in range(self.skip):
            if not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                states.append(state)
            else:
                states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(np.float32), reward, done, info

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        return states.astype(np.float32)


#def create_train_env(world, stage, action_type, output_path=None):
def create_train_env(layout, output_path=None):
    #env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))
    #env = gym_pacman.make("PacmanBerkeley-{}-{}-v0".format(world, stage))
    print("Create train env:",layout)
    env = gym.make('BerkeleyPacman-v0')
    #output_path = 'output/un_video.mp4' # Beware! Can freeze training for some reason.
    if output_path:
        #monitor = Monitor(256, 240, output_path)
        monitor = Monitor(150, 150, output_path)
    else:
        monitor = None

    # Pacman Actions https://github.com/Kautenja/nes-py/wiki/Wrap
    actions = ['North', 'South', 'East', 'West', 'Stop']
    #actions = [['up'],['down'],['right'],['left'],['NOOP']]
    #env = BinarySpaceToDiscreteSpaceEnv(env, actions)
    #env = JoypadSpace(env, actions)
    #env = CustomReward(env, monitor)
    env = SameReward(env, monitor)
    #env = CustomSkipFrame(env)
    #env = DQNSkipFrame(env)
    #env = SimpleSkipFrame(env)
    #env = MinimSkipFrame(env)
    # Four times same frame input, no skip
    #env = NoSkipFrame(env)
    # Four rotations of same frame input, no skip
    env = NoSkipFrameFourRotations(env)
    #return env, env.observation_space.shape[0], len(actions)
    num_inputs_to_nn = 4#x84x84
    num_outputs_from_nn = len(actions)
    return env, num_inputs_to_nn, num_outputs_from_nn
