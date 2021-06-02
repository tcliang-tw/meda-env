import copy
import math
import queue
import random
import numpy as np
from PIL import Image
from enum import IntEnum
from datetime import datetime

import gym
from gym import error, spaces, utils
from gym.utils import seeding

class Action(IntEnum):
    N = 0 #North
    E = 1 #East
    S = 2 #South
    W = 3 #West
    NE = 4
    SE = 5
    SW = 6
    NW = 7

class Droplet:
    def __init__(self, x_min, x_max, y_min, y_max):
        if x_min > x_max or y_min > y_max:
            raise TypeError('Droplet() inputs are illegal')
        self.x_min = int(x_min)
        self.x_max = int(x_max)
        self.y_min = int(y_min)
        self.y_max = int(y_max)
        self.x_center = int((x_min + x_max) / 2)
        self.y_center = int((y_min + y_max) / 2)

    def __repr__(self):
        return "Droplet(x = " + str(self.x_center) + ", y = " +\
                str(self.y_center) + ", r = " +\
                str(self.x_max - self.x_center) + ")"

    def __eq__(self, rhs):
        if isinstance(rhs, Droplet):
            return self.x_min == rhs.x_min and self.x_max == rhs.x_max and\
                    self.y_min == rhs.y_min and self.y_max == rhs.y_max and\
                    self.x_center == rhs.x_center and\
                    self.y_center == rhs.y_center
        else:
            return False

    def isPointInside(self, point):
        ''' point is in the form of (y, x) '''
        if point[0] >= self.y_min and point[0] <= self.y_max and\
                point[1] >= self.x_min and point[1] <= self.x_max:
            return True
        else:
            return False

    def isDropletOverlap(self, m):
        if self._isLinesOverlap(self.x_min, self.x_max, m.x_min, m.x_max) and\
                self._isLinesOverlap(self.y_min, self.y_max, m.y_min, m.y_max):
            return True
        else:
            return False

    def isTooClose(self, m):
        """ Return true if distance is less than 1.2 radius sum """
        distance = self.getDistance(m)
        r1 = self.x_max - self.x_center
        r2 = m.x_max - m.x_center
        return distance < 1.5 * (r1 + r2 + 1)

    def _isLinesOverlap(self, xa_1, xa_2, xb_1, xb_2):
        if xa_1 > xb_2:
            return False
        elif xb_1 > xa_2:
            return False
        else:
            return True

    def getDistance(self, droplet):
        delta_x = self.x_center - droplet.x_center
        delta_y = self.y_center - droplet.y_center
        return math.sqrt(delta_x * delta_x + delta_y * delta_y)

    def shiftX(self, step):
        self.x_min += step
        self.x_max += step
        self.x_center += step

    def shiftY(self, step):
        self.y_min += step
        self.y_max += step
        self.y_center += step

    def move(self, action, width, length):
        if action == Action.N:
            self.shiftY(-3)
        elif action == Action.E:
            self.shiftX(3)
        elif action == Action.S:
            self.shiftY(3)
        elif action == Action.W:
            self.shiftX(-3)
        elif action == Action.NE:
            self.shiftX(2)
            self.shiftY(-2)
        elif action == Action.SE:
            self.shiftX(2)
            self.shiftY(2)
        elif action == Action.SW:
            self.shiftX(-2)
            self.shiftY(2)
        elif action == Action.NW:
            self.shiftX(-2)
            self.shiftY(-2)
        if self.x_max >= length:
            self.shiftX(length - 1 - self.x_max)
        elif self.x_min < 0:
            self.shiftX(0 - self.x_min)
        if self.y_max >= width:
            self.shiftY(width - 1 - self.y_max)
        elif self.y_min < 0:
            self.shiftY(0 - self.y_min)

class RoutingTaskManager:
    def __init__(self, w, l, n_droplets):
        self.width = w
        self.length = l
        self.n_droplets = n_droplets
        self.starts = []
        self.droplets = []
        self.destinations = []
        self.distances = []
        self.n_limit = int(w / 15) * int(l / 15)
        if n_droplets > self.n_limit:
            raise RuntimeError("Too many droplets in the " + str(w) + "x" +\
                    str(l) + " MEDA array")
        random.seed(datetime.now())
        for i in range(n_droplets):
            self.addTask()

    def refresh(self):
        self.starts.clear()
        self.droplets.clear()
        self.destinations.clear()
        self.distances.clear()
        for i in range(self.n_droplets):
            self.addTask()

    def restart(self):
        self.droplets = copy.deepcopy(self.starts)
        self._updateDistances()

    def addTask(self):
        if len(self.droplets) >= self.n_limit:
            return
        self._genLegalDroplet(self.droplets)
        self._genLegalDroplet(self.destinations)
        while(self.droplets[-1].isDropletOverlap(self.destinations[-1])):
            self.destinations.pop()
            self._genLegalDroplet(self.destinations)
        self.distances.append(
                self.droplets[-1].getDistance(self.destinations[-1]))
        self.starts.append(copy.deepcopy(self.droplets[-1]))

    def _genLegalDroplet(self, dtype):
        d_center = self.getRandomYX()
        new_d = Droplet(d_center[1] - 3, d_center[1] + 3, d_center[0] - 3,
                d_center[0] + 3)
        while(not self._isGoodDroplet(new_d, dtype)):
            d_center = self.getRandomYX()
            new_d = Droplet(d_center[1] - 3, d_center[1] + 3, d_center[0] - 3,
                    d_center[0] + 3)
        dtype.append(new_d)

    def getRandomYX(self):
        return (random.randint(3, self.width - 4),
                random.randint(3, self.length - 4))

    def _isGoodDroplet(self, new_d, dtype):
        for d in dtype:
            if(d.isTooClose(new_d)):
                return False
        return True

    def _updateDistances(self):
        dist = []
        for drp, dst in zip(self.droplets, self.destinations):
            dist.append(drp.getDistance(dst))
        self.distances = dist

    def moveDroplets(self, actions, m_health):
        if len(actions) != self.n_droplets:
            raise RuntimeError("The number of actions is not the same as n_droplets")
        n_active_d = len(self.droplets)
        droplets = []
        destinations = []
        rewards = []
        #print(self.droplets)
        for i in range(n_active_d):
            goal_dist = self.droplets[i].x_max - self.droplets[i].x_center +\
                    self.destinations[i].x_max - self.destinations[i].x_center
            if self.distances[i] < goal_dist: # already achieved goal
                droplets.append(self.destinations[i])
                destinations.append(self.destinations[i])
                rewards.append(0.0)
            else:
                prob = self.getMoveProb(self.droplets[i], m_health)
                if random.random() <= prob:
                    self.droplets[i].move(actions[i], self.width, self.length)
                new_dist = self.droplets[i].getDistance(self.destinations[i])
                if new_dist < goal_dist: # get to the destination
                    rewards.append(1.0)
                elif new_dist < self.distances[i]: # closer to the destination
                    rewards.append(-0.05)
                else:
                    rewards.append(-0.1) # penalty for taking one step
                droplets.append(self.droplets[i])
                destinations.append(self.destinations[i])
        self.droplets = droplets
        self.destinations = destinations
        self._updateDistances()
        if rewards.count(1.0) == 0:
            rewards = self.updateRewardsBasedOnDist(rewards)
        return np.average(rewards)

    def getMoveProb(self, droplet, m_health):
        count = 0
        prob = 0.0
        for y in range(droplet.y_min, droplet.y_max + 1):
            for x in range(droplet.x_min, droplet.x_max + 1):
                prob += m_health[y][x]
                count += 1
        return prob / float(count)

    def updateRewardsBasedOnDist(self, rewards):
        for i in range(len(self.droplets)):
            for j in range(i + 1, len(self.droplets)):
                safe_dst = self.droplets[i].x_max - self.droplets[i].x_center +\
                        self.droplets[j].x_max - self.droplets[j].x_center
                real_dst = self.droplets[i].getDistance(self.droplets[j])
                if real_dst < 1.5 * safe_dst:
                    rewards[i] -= 0.8
                    rewards[j] -= 0.8
                #elif real_dst < 2.0 * safe_dst:
                #    rewards[i] -= 0.1
                #    rewards[j] -= 0.1
        return rewards

    def getTaskStatus(self):
        goal_distances = []
        for drp, dst in zip(self.droplets, self.destinations):
            goal_distances.append(drp.x_max - drp.x_center + dst.x_max -\
                    dst.x_center)
        return [dist < gd for dist, gd in zip (self.distances, goal_distances)]

class BaseLineRouter:
    def __init__(self, w, l):
        self.width = w
        self.length = l

    def getEstimatedReward(self, routing_manager, m_health = None):
        road_map = []
        trajectories = []
        max_step = 0
        for drp, dest in zip(routing_manager.droplets,
                routing_manager.destinations):
            actions = self.addPath(road_map, drp, dest)
            trajectories.append(actions)
            if len(actions) > max_step:
                max_step = len(actions)
        for i, actions in enumerate(trajectories):
            l = len(actions)
            if l < max_step:
                trajectories[i] += [Action.N] * (max_step - l)
        rewards = []
        probs = []
        for i in range(max_step):
            if m_health:
                prob = 0.0
                for drp in routing_manager.droplets:
                    prob += routing_manager.getMoveProb(drp, m_health)
                prob /= len(routing_manager.droplets)
                probs.append(prob)
            actions_by_droplets = [actions[i] for actions in trajectories]
            r = routing_manager.moveDroplets(actions_by_droplets,
                    np.ones((self.width, self.length)))
            rewards.append(r)
        routing_manager.restart() # this is important to revert the game
        if m_health is None:
            return sum(rewards), max_step
        else:
            mod_reward = 0.0
            for p, r in zip(probs, rewards):
                mod_reward += r / p
            max_step = max_step / np.average(probs)
            return mod_reward, max_step

    def markLocation(self, road_map, drp, value):
        for y in range(drp.y_min, drp.y_max + 1):
            for x in range(drp.x_min, drp.x_max + 1):
                road_map[y][x] = value

    def addPath(self, road_map, drp, dest):
        actions = []
        delta_x = dest.x_center - drp.x_center
        delta_y = dest.y_center - drp.y_center
        if delta_x > 0:
            x_moves = [Action.E] * int(delta_x / 3)
        else:
            x_moves = [Action.W] * int(abs(delta_x) / 3)
        if delta_y > 0:
            y_moves = [Action.S] * int(delta_y / 3)
        else:
            y_moves = [Action.N] * int(abs(delta_y) / 3)
        for i in range(len(x_moves)):
            path = x_moves[:i] + y_moves + x_moves[i:]
            valid_path = True
            temp_drp = copy.deepcopy(drp)
            for i, act in enumerate(path):
                next_drp = copy.deepcopy(temp_drp)
                next_drp.move(act, self.width, self.length)
                if self.checkValidMove(next_drp, temp_drp, road_map, i + 1):
                    temp_drp = next_drp
                else:
                    valid_path = False
                    break
            if valid_path:
                actions = path
                break
        if len(actions) == 0:
            if len(y_moves) > 0:
                i = random.choice(range(len(y_moves)))
                action = y_moves[:i] + x_moves + y_moves[i:]
            else:
                action = x_moves
        this_map = np.full((self.width, self.length), -1)
        move_drp = copy.deepcopy(drp)
        for step, act in enumerate(actions):
            self.markLocation(this_map, move_drp, step)
            move_drp.move(act, self.width, self.length)
        self.markLocation(this_map, move_drp, len(actions))
        road_map.append(this_map)
        return actions

    def getScanArea(self, next_drp, prev_drp):
        points = set([])
        for y in range(next_drp.y_min, next_drp.y_max + 1):
            for x in range(next_drp.x_min, next_drp.x_max + 1):
                points.add((y, x))
        for y in range(prev_drp.y_min, prev_drp.y_max + 1):
            for x in range(prev_drp.x_min, prev_drp.x_max + 1):
                points.discard((y, x))
        return list(points)

    def checkValidMove(self, next_drp, prev_drp, road_map, next_v):
        scan_area = self.getScanArea(next_drp, prev_drp)
        for y, x in scan_area:
            for r_map in road_map:
                if r_map[y][x] >= next_v - 1 and r_map[y][x] <= next_v + 1:
                    return False
        return True

class MEDAEnv(gym.Env):
    """ A MEDA biochip environment
        [0,0]
          +---l---+-> x
          w       |
          +-------+
          |     [1,2]
          V
          y
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, w, l, n_droplets, b_degrade = False, per_degrade = 0.7):
        super(MEDAEnv, self).__init__()
        assert w > 0 and l > 0
        assert n_droplets > 0
        # OpenAI Gym setup
        self.actions = Action
        self.action_space = spaces.Discrete(
                int(math.pow(len(self.actions), n_droplets)))
        self.observation_space = spaces.Box(low = 0, high = 1,
                shape = (w, l, 3 * n_droplets), dtype = 'uint8')
        self.reward_range = (-1.0, 1.0)
        # Other data members
        self.width = w
        self.length = l
        self.routing_manager = RoutingTaskManager(w, l, n_droplets)
        self.b_degrade = b_degrade
        self.max_step = (w + l)
        self.m_health = np.ones((w, l))
        self.m_usage = np.zeros((w, l))
        if b_degrade:
            self.m_degrade = np.random.rand(w, l)
            self.m_degrade = self.m_degrade * 0.4 + 0.6
            selection = np.random.rand(w, l)
            per_healthy = 1. - per_degrade
            self.m_degrade[selection < per_healthy] = 1.0
        else:
            self.m_degrade = np.ones((w, l))
        # variables below change every game
        self.step_count = 0

    def step(self, actions):
        self.step_count += 1
        l_actions = self.decompressedAction(actions)
        reward = self.routing_manager.moveDroplets(l_actions, self.m_health)
        obs = self.getObs()
        if self.step_count <= self.max_step:
            status = self.routing_manager.getTaskStatus()
            done = np.all(status)
            self.addUsage(status)
        else:
            done = True
        return obs, reward, done, {}

    def decompressedAction(self, actions):
        l_actions = []
        while actions > 0:
            l_actions.append(Action(actions % len(self.actions)))
            actions = int(actions / len(self.actions))
        while len(l_actions) < self.routing_manager.n_droplets:
            l_actions.append(Action(0))
        return l_actions

    def reset(self):
        self.step_count = 0
        self.routing_manager.refresh()
        obs = self.getObs()
        self._updateHealth()
        return obs

    def render(self, mode = 'human'):
        """ Show environment """
        #goal:2, pos:1, degrade: -1
        if mode == 'human':
            img = np.zeros(shape = (self.width, self.length))
            for drp, des in zip(self.routing_manager.droplets,
                    self.routing_manager.destinations):
                for y in range(des.y_min, des.y_max + 1):
                    for x in range(des.x_min, des.x_max + 1):
                        img[y][x] = 2
                for y in range(drp.y_min, drp.y_max + 1):
                    for x in range(drp.x_min, drp.x_max + 1):
                        img[y][x] = 1
            if self.b_degrade:
                img[self.m_health < 0.5] = -1
            return img
        elif mode == 'rgb_array':
            # Default is grey
            img = np.full((self.width, self.length, 3), 192, dtype = np.uint8)
            for des in self.routing_manager.destinations:
                for y in range(des.y_min, des.y_max + 1):
                    for x in range(des.x_min, des.x_max + 1):
                        img[y][x] = [0, 255, 0] # green for destinations
            for drp in self.routing_manager.droplets:
                for y in range(drp.y_min, drp.y_max + 1):
                    for x in range(drp.x_min, drp.x_max + 1):
                        img[y][x] = [0, 0, 255] # blue for droplets
            if self.b_degrade:
                img[self.m_health < 0.5] = [255, 102, 255] #light purple
                img[self.m_health < 0.7] = [255, 153, 255] #purple
            return img
        else:
            raise RuntimeError('Unknown mode in render')

    def close(self):
        """ close render view """
        pass

    def printHealthSatus(self):
        print('### Env Health ###')
        n_bad = np.count_nonzero(self.m_health < 0.2)
        n_mid = np.count_nonzero(self.m_health < 0.5)
        n_ok = np.count_nonzero(self.m_health < 0.8)
        print('Really bad:', n_bad,
                'Halfly degraded:', n_mid - n_bad,
                'Mildly degraded', n_ok - n_mid)

    def addUsage(self, status):
        for i in range(self.routing_manager.n_droplets):
            if not status[i]:
                droplet = self.routing_manager.droplets[i]
                for y in range(droplet.y_min, droplet.y_max + 1):
                    for x in range(droplet.x_min, droplet.x_max + 1):
                        self.m_usage[y][x] += 1

    def _updateHealth(self):
        if not self.b_degrade:
            return
        index = self.m_usage > 50.0 #degrade here
        self.m_health[index] = self.m_health[index] * self.m_degrade[index]
        self.m_usage[index] = 0

    def getObs(self):
        """
        RGB format of image
        Obstacles - red in layer 0
        Goal      - greed in layer 1
        Droplet   - blue in layer 2
        """
        obs = np.zeros(shape = (self.width, self.length,
                3 * self.routing_manager.n_droplets))
        for i in range(self.routing_manager.n_droplets):
            # First add other droplets in 3 x i layer
            for j in range(self.routing_manager.n_droplets):
                if j == i:
                    continue
                o_drp = self.routing_manager.droplets[j]
                obs = self._addDropletInObsLayer(obs, o_drp, 3 * i)
            # Add destination in 3 x i + 1 layer
            dst = self.routing_manager.destinations[i]
            obs = self._addDropletInObsLayer(obs, dst, 3 * i + 1)
            # Add droplet in 3 x i + 2 layer
            drp = self.routing_manager.droplets[i]
            obs = self._addDropletInObsLayer(obs, drp, 3 * i + 2)
        return obs

    def _addDropletInObsLayer(self, obs, droplet, layer):
        for y in range(droplet.y_min, droplet.y_max + 1):
            for x in range(droplet.x_min, droplet.x_max + 1):
                obs[y][x][layer] = 1
        return obs

