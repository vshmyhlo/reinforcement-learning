import math

import numpy as np
from gym_minigrid.minigrid import *
from gym_minigrid.register import register

# TODO: can it learn to go as far as possible from initial position?


class FourRoomsCustom(MiniGridEnv):
    def __init__(self, agent_pos=None, goal_pos=None):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos

        self.n_max = 15000 / 16
        self.n = -1

        super().__init__(grid_size=19, max_steps=100)

    def reset(self):
        self.n += 1

        output = super().reset()
        self.visited = set()
        # self.dist_before = 0

        return output

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            # dif = min(self.n / self.n_max, 1)
            #
            # w = self.grid.width * 2
            # h = self.grid.height * 2
            #
            # w = 3 + dif * (w - 3)
            # h = 3 + dif * (h - 3)
            #
            # top = (self.agent_pos[0] - w / 2, self.agent_pos[1] - h / 2)
            # size = (w, h)
            #
            # top = tuple(round(x) for x in top)
            # size = tuple(round(x) for x in size)

            # self.goal_pos = self.place_obj(Goal(), top=top, size=size)

            self.goal_pos = self.place_obj(Goal())

        self.mission = "Reach the goal"

    def step(self, action):
        pos_before = self.agent_pos
        obs, reward, done, info = MiniGridEnv.step(self, action)
        pos_after = self.agent_pos

        # init
        reward_extra = reward * 10 - reward

        # panalize if failed
        if done and reward == 0:
            reward_extra -= 1

        # reward if closer to goal
        reward_extra += dist(pos_before, self.goal_pos) - dist(pos_after, self.goal_pos)

        # penalise for extra steps
        reward_extra -= 0.01

        # penalise for revisiting states
        if tuple(pos_after) in self.visited:
            reward_extra -= 0.1
        else:
            self.visited.add(tuple(pos_after))
            floor = Floor()
            self.put_obj(floor, *pos_after)

        # final reward
        dif = 1 - min(self.n / self.n_max, 1)
        reward = reward + dif * reward_extra

        return obs, reward, done, info


def dist(av, bv):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(av, bv)))


def eq(av, bv):
    return all(a == b for a, b in zip(av, bv))


register(
    id="MiniGrid-FourRooms-Custom-v0",
    entry_point="envs:FourRoomsCustom",
)
