import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from .graphicsDisplay import PacmanGraphics, FirstPersonPacmanGraphics, DEFAULT_GRID_SIZE

from .game import Actions
from .pacman import ClassicGameRules
from .layout import getLayout, getRandomLayout

from .ghostAgents import DirectionalGhost
from .pacmanAgents import OpenAIAgent

from gym.utils import seeding

import json
import os

DEFAULT_GHOST_TYPE = 'DirectionalGhost'

MAX_GHOSTS = 5

PACMAN_ACTIONS = ['North', 'South', 'East', 'West', 'Stop']
PACMAN_DIRECTIONS = ['North', 'South', 'East', 'West']
ROTATION_ANGLES = [0, 180, 90, 270]

MAX_EP_LENGTH = 10000
PENALTY_MAX_EP = 500.0
PENALTY_ILLEGAL_ACTION = 10.0
PENALTY_TIME_STEP = 1.0

import os
fdir = '/'.join(os.path.split(__file__)[:-1])
#print(fdir)
layout_params = json.load(open(fdir + '/../../layout_params.json'))

#print("Layout parameters")
#print(layout_params)
#print("------------------")
#for k in layout_params:
#    print(k, ":",layout_params[k])
#print("------------------")

class PacmanEnv(gym.Env):
    layouts = [
        'capsuleClassic', 'contestClassic', 'mediumClassic', 'mediumGrid', 'minimaxClassic', 'openClassic', 'originalClassic', 'smallClassic', 'capsuleClassic', 'smallGrid', 'testClassic', 'trappedClassic', 'trickyClassic'
    ]

    noGhost_layouts = [l + '_noGhosts' for l in layouts]

    MAX_MAZE_SIZE = (7, 7)
    num_envs = 1

    observation_space = spaces.Box(low=0, high=255,
                         shape=(84, 84, 3), dtype=np.uint8)

    def __init__(self):
        self.action_space = spaces.Discrete(4) # up, down, left, right
        self.display = PacmanGraphics(1.0)#PacmanGraphics(1.0)
        self._action_set = range(len(PACMAN_ACTIONS))
        self.location = None
        self.viewer = None
        self.done = False
        self.layout = None
        self.np_random = None
        self.np_random = np.random

    def setObservationSpace(self):
        screen_width, screen_height = self.display.calculate_screen_dimensions(self.layout.width, self.layout.height)
        #print("Layout:", self.layout.width, "x" ,self.layout.height)
        #print("Screen size:", screen_width, "x" ,screen_height)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(int(screen_height),
                                                   int(screen_width),
                                                   3), dtype=np.uint8)
    def level1(self):
        layouts = []
        
    def chooseLayout(self, randomLayout=False,
        chosenLayout=None, no_ghosts=False):
        randomLayout=True
        chosenLayout="mediumClassic_openGhostHome"
        #print("Chosen lay:", chosenLayout)
        if not chosenLayout:
            #level1
            layouts = ['microGrid_superEasy1','microGrid_superEasy2']
            chosenLayout = np.random.choice(layouts)
    #         if np.random.random() > 0.5:
    #             chosenLayout='microGrid_superEasy1'
    #         else:
    #             chosenLayout='microGrid_superEasy2'
            #level2
            layouts = ['smallGrid_superEasy1','smallGrid_superEasy2']
            chosenLayout=np.random.choice(layouts)
    #         if np.random.random() > 0.5:
    #             chosenLayout='smallGrid_superEasy1'
    #         else:
    #             chosenLayout='smallGrid_superEasy2'
            randomLayout = False
            chosenLayout='smallGrid_moreFood'
        #print("random layout:",randomLayout)
        if randomLayout:
            print("Random layout generated")
            self.layout = getRandomLayout(layout_params, self.np_random)
        else:
            if chosenLayout is None:
                if not no_ghosts:
                    chosenLayout = self.np_random.choice(self.layouts)
                else:
                    chosenLayout = self.np_random.choice(self.noGhost_layouts)
            self.chosen_layout = chosenLayout
            #print("Chosen layout", chosenLayout)
            self.layout = getLayout(chosenLayout)
        self.maze_size = (self.layout.width, self.layout.height)

    def seed(self, seed=None):
        if self.np_random is None:
            self.np_random, seed = seeding.np_random(seed)
        self.chooseLayout(randomLayout=False)
        return [seed]

    def reset(self, layout=None):
        # get new layout
        #print("Reset lay:", layout, self.layout)
        
        if layout is None:
            layout = self.chooseLayout(randomLayout=True)
        #self.chooseLayout(randomLayout=False)

        self.step_counter = 0
        self.cum_reward = 0
        self.done = False

        self.setObservationSpace()

        # we don't want super powerful ghosts. Default: 0.2/0.2
        random_prob_atack = np.random.uniform(0.0, 0.9)
        self.ghosts = [DirectionalGhost( i+1, prob_attack=random_prob_atack, prob_scaredFlee=0.2) for i in range(MAX_GHOSTS)]

        # this agent is just a placeholder for graphics to work
        self.pacman = OpenAIAgent()

        self.rules = ClassicGameRules(300)
        self.rules.quiet = False

        self.game = self.rules.newGame(self.layout, self.pacman, self.ghosts,
            self.display, False, False)

        self.game.init()

        self.display.initialize(self.game.state.data)
        self.display.updateView()

        self.location = self.game.state.data.agentStates[0].getPosition()
        self.ghostLocations = [a.getPosition() for a in self.game.state.data.agentStates[1:]]
        self.ghostInFrame = any([np.sum(np.abs(np.array(g) - np.array(self.location))) <= 2 for g in self.ghostLocations])

        self.location_history = [self.location]
        self.orientation = PACMAN_DIRECTIONS.index(self.game.state.data.agentStates[0].getDirection())
        self.orientation_history = [self.orientation]
        self.illegal_move_counter = 0

        self.cum_reward = 0

        self.initial_info = {
            'past_loc': [self.location_history[-1]],
            'curr_loc': [self.location_history[-1]],
            'past_orientation': [[self.orientation_history[-1]]],
            'curr_orientation': [[self.orientation_history[-1]]],
            'illegal_move_counter': [self.illegal_move_counter],
            'ghost_positions': [self.ghostLocations],
            'ghost_in_frame': [self.ghostInFrame],
            'step_counter': [[0]],
        }

        return self._get_image()

    def step(self, action):
        self.cum_reward -= PENALTY_TIME_STEP
        # implement code here to take an action
        if self.step_counter >= MAX_EP_LENGTH or self.done:
            self.step_counter += 1
            if not self.done:
                # max episode len reached, penalty!
                self.cum_reward -= PENALTY_MAX_EP
                print("Max episode length reached. Reward:", self.cum_reward)
            else:
                print("Done. Reward:", self.cum_reward)
            return np.zeros(self.observation_space.shape), 0.0, True, {
                'past_loc': [self.location_history[-2]],
                'curr_loc': [self.location_history[-1]],
                'past_orientation': [[self.orientation_history[-2]]],
                'curr_orientation': [[self.orientation_history[-1]]],
                'illegal_move_counter': [self.illegal_move_counter],
                'step_counter': [[self.step_counter]],
                'ghost_positions': [self.ghostLocations],
                'r': [self.cum_reward],
                'l': [self.step_counter],
                'ghost_in_frame': [self.ghostInFrame],
                'episode': [{
                    'r': self.cum_reward,
                    'l': self.step_counter
                }],
                'max_ep': self.step_counter >= MAX_EP_LENGTH
            }
        #print("Action chosen :", action, "  ", end="\r")

        
        #pacman_action = PACMAN_ACTIONS[action]
        pacman_action = PACMAN_ACTIONS[action]
        #print(self.step_counter, "\t", self.cum_reward,"\t", " Action chosen :", action, PACMAN_ACTIONS[action], end="\r")
        #      self.step_counter,self.cum_reward,"   ", end=" ")

        legal_actions = self.game.state.getLegalPacmanActions()
        illegal_action = False
        if pacman_action not in legal_actions:
            self.illegal_move_counter += 1
            illegal_action = True
            pacman_action = 'Stop' # Stop is always legal

        reward = self.game.step(pacman_action)
        # reward shaping for illegal actions
        if illegal_action:
            reward -= PENALTY_ILLEGAL_ACTION

        self.cum_reward += reward

        done = self.game.state.isWin() or self.game.state.isLose()

        self.location = self.game.state.data.agentStates[0].getPosition()
        self.location_history.append(self.location)
        self.ghostLocations = [a.getPosition() for a in self.game.state.data.agentStates[1:]]

        self.orientation = PACMAN_DIRECTIONS.index(self.game.state.data.agentStates[0].getDirection())
        self.orientation_history.append(self.orientation)

        extent = (self.location[0] - 1, self.location[1] - 1),(self.location[0] + 1, self.location[1] + 1),
        self.ghostInFrame = any([ g[0] >= extent[0][0] and g[1] >= extent[0][1] and g[0] <= extent[1][0] and g[1] <= extent[1][1]
                                    for g in self.ghostLocations])

        self.step_counter += 1

        if self.step_counter >= MAX_EP_LENGTH:
            if not self.done:
                # max episode len reached, penalty!
                self.cum_reward -= PENALTY_MAX_EP
                reward -= PENALTY_MAX_EP

        info = {
            'past_loc': [self.location_history[-2]],
            'curr_loc': [self.location_history[-1]],
            'past_orientation': [[self.orientation_history[-2]]],
            'curr_orientation': [[self.orientation_history[-1]]],
            'illegal_move_counter': [self.illegal_move_counter],
            'step_counter': [[self.step_counter]],
            'episode': [None],
            'ghost_positions': [self.ghostLocations],
            'ghost_in_frame': [self.ghostInFrame],
            # adding score so far
            'score': self.cum_reward,
            'max_ep': self.step_counter >= MAX_EP_LENGTH
        }

        if self.step_counter >= MAX_EP_LENGTH:
            print("Max episode length reached. Reward:", self.cum_reward)
            done = True

        self.done = done

        if self.done: # only if done, send 'episode' info
            #print("Done. Reward:", self.cum_reward)
            info['episode'] = [{
                'r': self.cum_reward,
                'l': self.step_counter
            }]
        return self._get_image(), reward, done, info

    def get_action_meanings(self):
        return [PACMAN_ACTIONS[i] for i in self._action_set]

    # just change the get image function
    def _get_image(self):
        # get x, y
        image = self.display.image
        #background = Image.new('RGB', image.size, color)
        # For partial observable environment (crop 'outside view' range)
        w, h = image.size
        DEFAULT_GRID_SIZE_X, DEFAULT_GRID_SIZE_Y = w / float(self.layout.width), h / float(self.layout.height)
        OBS_RANGE_X=1 # observed cells at each side
        OBS_RANGE_Y=1
        extent = [
            DEFAULT_GRID_SIZE_X *  (self.location[0] - OBS_RANGE_X),
            DEFAULT_GRID_SIZE_Y *  (self.layout.height - (self.location[1]+1 + OBS_RANGE_Y+0.2)), # VISION RANGE
            DEFAULT_GRID_SIZE_X *  (self.location[0]+1 + OBS_RANGE_X),
            DEFAULT_GRID_SIZE_Y *  (self.layout.height - (self.location[1] - OBS_RANGE_Y-0.2))]
        extent = tuple([int(e) for e in extent])
        self.image_sz = (84, 84)
        #image = image.crop(extent).resize(self.image_sz)
        image = image.resize(self.image_sz)
        return np.array(image)[:,:,:3] # Remove 4th channel if there is one

    def render(self, mode='rgb_array'):
        # Mode:
        # human : show input to network as window
        # rgb_array: hide input to network
        img = self._get_image()
        #has_alpha_channel = True
        #if has_alpha_channel:
            # Remove 4th channel
        #    img = "img[:,:,:2]"
        mode = 'human'
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            #if self.viewer is None:
            #    from gym.envs.classic_control import rendering
            #    self.viewer = rendering.Viewer(500,500)
            #    self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
             #   rod = rendering.make_capsule(1, .2)
            #    rod.set_color(.8, .3, .3)
            #    self.img = rendering.SimpleImageViewer(img)
            #    #self.imgtrans = rendering.Transform()
            #    #self.img.add_attr(self.imgtrans)
            #    self.viewer.add_geom(self.img)
            #mode = 'human'
            #self.viewer.render(return_rgb_array = mode=='rgb_array') 
            return self.viewer.isopen

    def close(self):
        # TODO: implement code here to do closing stuff
        if self.viewer is not None:
            self.viewer.close()
        self.display.finish()

    def __del__(self):
        self.close()
