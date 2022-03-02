import os
import math
import copy
from typing import NamedTuple, Dict
from sort_utils import sort_str_num
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
import collections
import time
from random import uniform
import random
from os.path import join
import dm_env
from acme import specs, types
# import tensorflow as tf

# Define some colours
green = (0, 255, 50)
light_blue = (0, 255, 255)
blue = (0, 0, 255)
red = (255, 0, 0)
yellow = (255, 255, 0)
white = (255, 255, 255)

# Max ball speed
max_ball_speed = 5.0

# R G B
field_col = (0, 200, 50)

own_goal = (255, 140, 0)
opp_goal = (128, 0, 128)

col_list = [None, white, blue, red, own_goal, opp_goal, light_blue]

ray_i_dict = {"dist": 0, "ball": 1, "teammate": 2, "opponent": 3, "own_goal": 4, "opp_goal": 5, "field": 6}

ray_spec_dict = {"dist": [0, 1.0], "ball": [0, 1], "teammate": [0, 1], "opponent": [0, 1], "own_goal": [0, 1], 
"opp_goal": [0, 1], "field": [0, 1], "time": [0, 1]}
init_position = [
            # Goalie
            (0.0, -0.45),

            # Defenders
            (-0.225, -0.3),
            (-0.075, -0.3),
            (0.075, -0.3),
            (0.225, -0.3),

            # Midfielders
            (-0.2, -0.2),
            (0.0, -0.2),
            (0.2, -0.2),

            # Attackers
            (-0.2, -0.1),
            (0.0, -0.1),
            (0.2, -0.1),
        ]

def is_nn_bot(agent):
    agent_str = "<class 'CustomSoccerEnv.Agents.nn_agent.NNBot'>"
    return str(type(agent)) == agent_str or agent == "agent"

def rad_rot_to_xy(rad_rot):
    return math.cos(rad_rot), math.sin(rad_rot)

def cart2pol(x, y):
    radius = math.sqrt(x ** 2 + y ** 2)
    angle = math.atan2(y, x)
    return radius, angle

# Dummy class to get the observation in the correct format
class OLT(NamedTuple):
    """Container for (observation, legal_actions, terminal) tuples."""

    observation: types.Nest
    legal_actions: types.Nest
    terminal: types.Nest

def calc_line_ball_stats(angle, circ_rad, circ_cen, start_point=None, cen_point=None, calc_coords = False):
    assert start_point is not None or cen_point is not None
    sin_comp = math.sin(angle)
    cos_comp = math.cos(angle)
    x1 = cen_point[0] + 0.5*sin_comp - circ_cen[0]
    y1 = cen_point[1] - 0.5*cos_comp - circ_cen[1]
    x2 = cen_point[0] - 0.5*sin_comp - circ_cen[0]
    y2 = cen_point[1] + 0.5*cos_comp - circ_cen[1]

    D = x1 * y2 - x2 * y1
    delta = circ_rad*circ_rad - D * D
    if calc_coords:
        dx = x2 - x1
        dy = y2 - y1

        if delta >= 0:
            Ddy = D * dy
            min_Ddx = -D * dx
            comb_coord = [Ddy, min_Ddx]

            _, new_ang = cart2pol(comb_coord[0] + circ_cen[0] - start_point[0], comb_coord[1] + circ_cen[1] - start_point[1])
            new_ang = -new_ang + math.pi / 2

            if new_ang < 0:
                new_ang += 2*math.pi
            if abs(angle - new_ang) < math.pi/2:
                coord = [comb_coord[0] + circ_cen[0], comb_coord[1] + circ_cen[1]]
                coord_dist = dist(start_point, coord)

                return True, coord_dist
            else:
                return False, None
        else:
            return False, None
    else:
        return delta, cos_comp, sin_comp

def speed(vel):
    return math.sqrt(math.pow(vel[0], 2)+math.pow(vel[1], 2))

class Agent(object):
    def __init__(self, pos, r_to_s, team, name, max_dist, field_size, num_rays, vision_range, rotation=0.0, radius=1):

        self.set_team(team)
        self._max_dist = max_dist
        self.name = name
        self.dir_col = yellow
        self.pos = pos
        self.leg_length = 3 * radius
        self.rot = rotation
        self.radius = radius
        self.offset = [field_size[0]/2, field_size[1]/2]

        if r_to_s is not None:
            self.screen_rad = int(radius * r_to_s)
            self.r_to_s = r_to_s

        if num_rays < 3:
            raise ValueError("Num rays must be greater than 2.")
        
        if vision_range < 0.1:
            raise ValueError("Vision range must be greater than 0.1")

        self.num_rays = num_rays
        self.vision_range = vision_range

    def set_team(self, team):
        self.team = team.lower()
        if self.team=='blue':
            self.colour = blue
            self.team_num = 0
        else:
            self.colour = red
            self.team_num = 1

    def render(self, field_surf, field_size=None, vision_cone=None, state_view=None):
        # Plot agent
        pygame.draw.circle(field_surf, self.colour, [int((self.pos[0] + self.offset[0])*self.r_to_s), int((self.pos[1] + self.offset[1])*self.r_to_s)],
                           self.screen_rad)
        sin_comp = self.leg_length*math.sin(self.rot)
        cos_comp = self.leg_length*math.cos(self.rot)

        x1 = self.pos[0] + sin_comp  + self.offset[0]
        y1 = self.pos[1] - cos_comp  + self.offset[1]
        x2 = self.pos[0] - sin_comp  + self.offset[0]
        y2 = self.pos[1] + cos_comp  + self.offset[1]
        pygame.draw.line(field_surf, self.colour, (x1 * self.r_to_s - 1, y1 * self.r_to_s - 1),
                         (x2 * self.r_to_s - 1, y2 * self.r_to_s - 1), 2)

        sin_comp = self.radius * math.sin(self.rot)
        cos_comp = self.radius * math.cos(self.rot)
        x3 = self.pos[0] + cos_comp + self.offset[0]
        y3 = self.pos[1] + sin_comp + self.offset[1]
        pygame.draw.line(field_surf, self.dir_col, ((self.pos[0]+ self.offset[0]) * self.r_to_s, (self.pos[1]+ self.offset[1]) * self.r_to_s),
                         (x3 * self.r_to_s, y3 * self.r_to_s),
                         2)


class Ball(object):
    def __init__(self, pos, r_to_s, field_size, friction=0.8, velocity=[], radius=1, body_speed=0.6, leg_speed=4.0):
        if len(velocity) == 0:
            velocity = [0.0, 0.0]
        self.pos = [pos[0], pos[1]]
        self.vel = [velocity[0], velocity[1]]
        self.radius = radius
        self.offset = [field_size[0]/2, field_size[1]/2]

        self.fric = friction
        self.body_speed = body_speed
        self.leg_speed = leg_speed

        if r_to_s is not None:
            self.screen_rad = int(radius * r_to_s)
            self.r_to_s = r_to_s

    def render(self, field_surf):
        pygame.draw.circle(field_surf, white, [int((self.pos[0]+self.offset[0])*self.r_to_s), int((self.pos[1]+self.offset[1])*self.r_to_s)],
                           self.screen_rad)

    def check_hit(self, agents):
        for agent in agents.values():
            # Check for circle touch
            tot_dist = dist(self.pos, agent.pos)
            if tot_dist < self.radius + agent.radius:
                x_diff = self.pos[0] - agent.pos[0]
                y_diff = self.pos[1] - agent.pos[1]
                self.vel[0] += self.body_speed * x_diff / (tot_dist + 0.0001)
                self.vel[1] += self.body_speed * y_diff / (tot_dist + 0.0001)

            # Check for legs touch.
            if tot_dist < self.radius + agent.leg_length:
                delta, cos_comp, sin_comp = calc_line_ball_stats(agent.rot, self.radius, self.pos, cen_point=agent.pos)
                if delta >= 0:
                    if dist(self.pos, agent.pos) < 2 * agent.radius + 2 * self.radius:
                        self.vel[0] += self.leg_speed * cos_comp
                        self.vel[1] += self.leg_speed * sin_comp
                 
        # Check if is moving faster than the max ball speed.
        speed_val = speed(self.vel)
        if speed_val > max_ball_speed:
            reduction_ration = max_ball_speed/speed_val
            for i in range(2):
                self.vel[i] = self.vel[i]*reduction_ration

def dist(pt1, pt2):
    return math.sqrt(math.pow(pt2[0]-pt1[0], 2) + math.pow(pt2[1]-pt1[1], 2))

def calc_min_max_rot(has_hit, min_dist, min_hit_i, hit_i, cur_pos, circ_cen, circ_rad, start_rot, rot_inc, end_rot, num_rays):

    obj_dist = dist(cur_pos, circ_cen)

    delta_rot = math.atan(circ_rad/(obj_dist + 0.00001))
    the_cen = math.atan((circ_cen[1] - cur_pos[1])/(circ_cen[0] - cur_pos[0] + 0.00001))

    if circ_cen[0] - cur_pos[0] < 0:
        the_cen = math.pi + the_cen

    diff = the_cen - start_rot
    if diff > math.pi:
        the_cen -= 2*math.pi
    elif diff < -math.pi:
        the_cen += 2*math.pi

    the_start = the_cen - delta_rot
    the_end = the_cen + delta_rot

    if the_start <= end_rot and the_end >= start_rot:
        # Atleast one ray overlaps with view
        if start_rot <= the_start <= end_rot:
            start_i = int((the_start - start_rot) / rot_inc) + 1
        else:
            start_i = 0

        if start_rot <= the_end <= end_rot:
            end_i = int((the_end - start_rot) / rot_inc) + 1
        else:
            end_i = num_rays
       
        for i in range(start_i, end_i):
            if obj_dist < min_dist[i]:
                min_dist[i] = obj_dist
                has_hit[i] = True
                min_hit_i[i] = hit_i

def calc_closest_rot(vision_arr, hit_i, cur_pos, circ_cen, circ_rad, start_rot, rot_inc, end_rot, num_rays, max_dist, is_ball=False):
    obj_dist = dist(cur_pos, circ_cen)
    obj_cen_rot = math.atan2(circ_cen[1] - cur_pos[1], circ_cen[0] - cur_pos[0])
    diff = obj_cen_rot - start_rot

    # Get diff between 0 and 2*pi
    if diff < 0:
        obj_cen_rot += 2*math.pi

    if start_rot <= obj_cen_rot <= end_rot:
        cen_i = int(round((obj_cen_rot - start_rot) / rot_inc))
        obj_dist /= max_dist
        if obj_dist < vision_arr[cen_i, ray_i_dict["dist"]] or is_ball:
            vision_arr[cen_i, ray_i_dict["dist"]] = obj_dist
            vision_arr[cen_i, 1:] = 0.0
            vision_arr[cen_i, hit_i] = 1

class FootballEnv(object):
    def __init__(self, render_game=False, game_setting='xray_shared', players_per_team=[1, 0], game_diff=0.0, num_rays=91, vision_range=math.pi,
    horizontal_size=72, game_length=None, do_team_switch=True, include_wait=True, show_agent_rays=False, dom_rand_use_all=False, 
    add_touch_penalty=False, action_space="continuous", heatmap_save_loc=None, reset_setup="position"):

        # horizontal_size=110
        assert horizontal_size==72

        # Define sizes of the objects on the field
        self.field_size = (110, 76)
        self.in_field_size = (100, 70)  # Real football field roughly 100m by 70m.

        field_x = self.in_field_size[0]  # x_size, y_size
        field_y = self.in_field_size[1]
        assert field_x > field_y
        self._pos_norm = field_x / 2.0
        self._reset_setup = reset_setup

        self._add_touch_penalty = add_touch_penalty
        print("Pentalty is False! Make it true again.")

        self.num_horizontal_pixels = horizontal_size
        self.num_vertical_pixels = int(horizontal_size * self.in_field_size[1] / self.in_field_size[0])
        assert self.num_horizontal_pixels > self.num_vertical_pixels
        print("Make the agents and goal smaller.")
        self.goal_size = (3, 8*5)

        self._home_goal_reward = 0.0

        assert num_rays % 2 == 1
        self.num_rays = num_rays
        self.render_game = render_game
        self.dom_rand_use_all = dom_rand_use_all

        self.game_setting = game_setting

        self.x_out_start = -self.in_field_size[0] / 2.0
        self.y_out_start = -self.in_field_size[1] / 2.0

        self.x_out_end = self.in_field_size[0] / 2.0
        self.y_out_end = self.in_field_size[1] / 2.0

        self.x_goal1_start = self.x_out_start - self.goal_size[0]
        self.y_goal_start = - self.goal_size[1] / 2.0
        self.y_goal_end = self.goal_size[1] / 2.0

        self.x_goal2_start = self.in_field_size[0]/2

        self._game_results = collections.deque(maxlen=100)

        self._last_actions = {}

        self.rot_speed = 0.12*100/30
        self.move_speed = 1 #0.2*100/30

        if 0.0 <= game_diff <= 2.0:
            self._game_diff = game_diff
            self._field_diff = np.clip(game_diff, 0.0, 1.0)
        else:
            raise ValueError("Game defined outside normal ranges (0-2).")

        self.a_dict = {"move": 0, "rot": 1}

        if game_length is not None:
            self.game_type = "fixed"
            self.game_length = int(game_length)
        else:
            self.game_type = "dom_rand"
            
            # This last np.clip is just so that we can see the difficulty of the game after the 
            # max field size is reached.
            self.update_dom_rand_game_length()

        self.num_steps = 0
        assert action_space == "continuous" or action_space == "discrete"
        self._action_space = action_space
        self._max_dist = dist((0, 0), self.in_field_size)
        self.num_rays = num_rays
        self.vision_range = vision_range
        self.include_wait = include_wait

        # If the game is not being rendered then we should not include a wait step
        assert not (include_wait and not render_game)

        # For rendering
        self.focus_player = "agent_0"
        self.show_agent_rays = show_agent_rays
        self.r_to_s = (1080 / 1.5) / self.field_size[1]  # Real to screen scaling factor
        window_width = int(self.field_size[0] * self.r_to_s)
        window_height = int(self.field_size[1] * self.r_to_s)
        self.screen_size = (window_width, window_height)

        self.x_out_screen_start = (self.field_size[0]/2 + self.x_out_start) * self.r_to_s
        self.y_out_screen_start = (self.field_size[1]/2 + self.y_out_start) * self.r_to_s

        self.x_goal1_screen_start = (self.field_size[0]/2 + self.x_goal1_start) * self.r_to_s
        self.y_goal_screen_start = (self.field_size[1]/2 + self.y_goal_start) * self.r_to_s
        self.x_goal2_screen_start = (self.field_size[0]/2 + self.x_goal2_start) * self.r_to_s

        self.cen_screen_x = (self.field_size[0]/2) * self.r_to_s
        self.cen_screen_y_1 = (self.field_size[1]/2 + self.y_out_start) * self.r_to_s
        self.cen_screen_y_2 = (self.field_size[1]/2 + self.y_out_end) * self.r_to_s

        self.do_team_switch = do_team_switch

        self.team_field_sides = {"blue": "left", "red": "right"}

        if render_game:
            # initialize game engine
            os.environ['SDL_VIDEO_CENTERED'] = '1'
            pygame.init()
            self.clock = pygame.time.Clock()

            # Open a window
            self.screen = pygame.display.set_mode(self.screen_size)

            # Set title to the window
            pygame.display.set_caption("Football 2D")
            self._wait_period = 0.04 # 0.04 # 0.1 is the same as with RoboCup

        self.players_per_team = players_per_team
        self.num_players = players_per_team[0] + players_per_team[1]
        assert self.num_players % 2 == 0
        self.num_per_team = int(self.num_players/2)
        self.agent_keys = ["agent_"+str(i) for i in range(self.num_players)]

        # Save heatmap info
        self._heatmap_save_loc = heatmap_save_loc
        if self._heatmap_save_loc is not None:
            self._heatmap_save_loc = join(heatmap_save_loc, "heatmap.csv")
            with open(self._heatmap_save_loc, 'w') as fp:
                pass
        
        self.possible_agents = self.agent_keys
        self._discount = dict(zip(self.agent_keys, [np.float32(0.999)] * len(self.agent_keys)))
        self.goals = {"blue": 0, "red": 0}

        # Setup agents and balls
        team_list = {}
        for i in range(players_per_team[0]):
            team_list["agent_" + str(i)] = "blue"
        for i in range(players_per_team[0], self.num_players):
           team_list["agent_" + str(i)] = "red"

        # Create the agents
        self.agents = {}
        for agent_key in self.agent_keys:
            self.agents[agent_key] = Agent(pos=[0.0, 0.0], team=team_list[agent_key], name=agent_key,
                                           max_dist=self._max_dist, field_size=self.field_size,
                                     r_to_s=self.r_to_s, num_rays=num_rays, vision_range=vision_range, rotation=0.0)

        # Create the ball
        self.ball = Ball(pos=[0.0, 0.0], friction=0.9, field_size=self.field_size,
                         r_to_s=self.r_to_s)

        # Set obs and state calculator function
        if self.game_setting in ["ppo_attention_state", "hybrid_attention_state", "maddpg_attention_state"]:
            self._calc_obs_func = lambda a_key: self._calc_attention_obs(a_key)
            self._calc_state_func = lambda: self._calc_ppo_state_info()
        else:
            raise NotImplementedError("game_setting not implemented: ", self.game_setting)

        # Calculate the per agent specs
        per_agent_obs_spec = self._get_per_agent_obs_spec()
        per_agent_act_spec = self._get_per_agent_action_spec()
        per_agent_extra_spec = self._get_per_agent_extra_spec()

        self._observation_specs = {}
        self._action_specs = {}
        self._extra_specs = {"env_states": {}}
        for agent in self.agents:
            self._observation_specs[agent] = OLT(
                observation=per_agent_obs_spec,
                legal_actions={},
                terminal=specs.Array((1,), np.float32),
            )
            self._action_specs[agent] = per_agent_act_spec
            self._extra_specs["env_states"][agent] = per_agent_extra_spec

    @staticmethod
    def get_num_channels():
        return len(ray_i_dict.keys())

    def update_dom_rand_game_length(self):
        self._field_diff = np.clip(self._game_diff, 0.0, 1.0)
        self.game_length = int(50 + self._field_diff*350 + 100 * np.clip(self._game_diff-1, 0, 1))
    
    def _get_per_agent_obs_spec(self):
        if self.game_setting in ["xray_shared", "ppo_xray_state",  "xray_attention"]:
            obs_spec = self._get_xray_spec()
        elif self.game_setting in ["ppo_attention_state", "hybrid_attention_state", "maddpg_attention_state"]:
            # Feedforward specs
            player_obs_spec = specs.BoundedArray(
                shape=(7+5+11,),
                dtype="float32",
                name="ff_observation",
                minimum=np.array([0.0] + [-1.0] * 6 + [0.0] + [-1.0] * 4 + [0.0] * 11),
                maximum=np.array([1.0] + [1.0] * 6 + [1.0] + [1.0] * 4 + [1.0] * 11),
            )

            agent_obs_spec = specs.BoundedArray(
                shape=(7,), # + 11
                dtype="float32",
                name="ff_observation",
                minimum=np.array([0.0] + [-1.0] * 6), #  + [0.0] * 11
                maximum=np.array([1.0] + [1.0] * 6), #  + [1.0] * 11
            )
            obs_spec = [player_obs_spec] + [agent_obs_spec] * (self.num_players-1)
        else:
            raise NotImplementedError("This obs type has not been implemented yet: ", self.game_setting)
        return obs_spec

    def _get_per_agent_action_spec(self):
        if self._action_space == "continuous":
            act_min = [-1.0, -1.0]
            act_max = [1.0, 1.0]
            assert len(act_min) == len(act_max)
            self.action_size = len(act_min)
            action_spec = specs.BoundedArray(
                shape=(len(act_min),),
                dtype="float32",
                name="action",
                minimum=act_min,
                maximum=act_max,
            )
            return action_spec
        else:
            self.action_size = 9
            action_spec = specs.DiscreteArray(
                num_values=(9),
                dtype="int64",
                name="action",
            )
            return action_spec

    def _get_per_agent_extra_spec(self):
        if self.game_setting in ["ppo_xray_state", "ppo_attention_state", "hybrid_attention_state", "maddpg_attention_state"]:
            # Get specs
            # Time, b_x, b_y, b_v_x, b_v_y
            obs_min = np.array([0.0] + [-1.0]*4)
            obs_max = np.array([1.0] + [1.0]*4)
            assert obs_min.shape == obs_max.shape
            time_ball_state_spec = specs.BoundedArray(
                shape=obs_min.shape,
                dtype="float32",
                name="state_extra",
                minimum=obs_min,
                maximum=obs_max,
            )

            # p_x, p_y, rot_x, rot_y, a_x, a_y, position
            obs_min = np.array([-1.0]  * 6 + [0.0]*11)
            obs_max = np.array([1.0]  * 6 + [1.0]*11)
            assert obs_min.shape == obs_max.shape
            agent_state_spec = specs.BoundedArray(
                shape=obs_min.shape,
                dtype="float32",
                name="state_extra",
                minimum=obs_min,
                maximum=obs_max,
            )

            return [time_ball_state_spec] + [agent_state_spec] * self.num_players
        else:
            raise NotImplementedError("This obs type has not been implemented yet: ", self.game_setting)

    def block_to_true_coord(self, block_coords, agent_key):
        x_size = self.in_field_size[0]
        x_block_size = self.num_horizontal_pixels
        y_size = self.in_field_size[1]
        y_block_size = self.num_vertical_pixels
        true_x = (block_coords[0]/x_block_size - 0.5)*x_size
        true_y = (block_coords[1]/y_block_size - 0.5)*y_size
        if self.team_field_sides[self.agents[agent_key].team] == "right":
            # Invert the positions and velocities
            true_x *= -1.0
            true_y *= -1.0

        return (true_x, true_y)

    def _calc_attention_obs(self, focus_agent_key):
        focus_agent = self.agents[focus_agent_key]
        focus_team = focus_agent.team
        focus_last_action = self._last_actions[focus_agent_key]

        # Add the time left, agent position, rotation
        need_swap = self.team_field_sides[focus_team] == "right"
        player_obs = self._get_abs_obs(agent=focus_agent, need_swap=need_swap, last_action=focus_last_action)

        # Relative ball info
        player_obs.extend(self._get_rel_obs(focus_agent=focus_agent, target_object=self.ball))
        attention_obs = [np.array(player_obs, dtype=np.float32)]

        team_agents = []
        opponent_agents = []

        for i, other_agent_key in enumerate(self.agent_keys):
            agent = self.agents[other_agent_key]
            if agent != focus_agent:
                if agent.team == focus_team:
                    team_agents.append([agent, other_agent_key])
                else:
                    opponent_agents.append([agent, other_agent_key])

        # TODO: Maybe shuffle team_agents and opponent_agents?
        agent_list = team_agents + opponent_agents

        for i, agent_info in enumerate(agent_list):
            agent, other_agent_key = agent_info
            assert agent != focus_agent
            # Get other player info
            other_agent_obs = self._get_rel_obs(focus_agent=focus_agent, target_object=agent,
                                                last_action=self._last_actions[other_agent_key])

            if i < int(len(agent_list) / 2):
                assert focus_agent.team == agent.team
            else:
                assert focus_agent.team != agent.team
            attention_obs.append(np.array(other_agent_obs, dtype=np.float32))
        return attention_obs

    def get_one_hot_id(self, agent):
        # Note: This might not work if the agent id names change!
        one_hot = [0.0] * 11
        if self.num_players == 22:
            one_hot[int(agent.split("_")[-1])%self.num_per_team] = 1.0
        else:
            # Random embedding if agents are less than 22
            one_hot[random.randint(0, 10)] = 1.0
        return one_hot

    def _calc_ppo_state_info(self):
        focus_agent = self.agents["agent_0"]

        sign = 1.0
        if self.team_field_sides[focus_agent.team] == "right":
            sign = -1.0

        time_ball_obs = np.array([1.0 - float(self.num_steps) / self.game_length,
                        sign*self.ball.pos[0] / self._pos_norm, sign*self.ball.pos[1] / self._pos_norm,
                        sign*self.ball.vel[0] / max_ball_speed, sign*self.ball.vel[1] / max_ball_speed],
                                 dtype=np.float32)
        # Get agents
        blue_agents = []
        red_agents = []
        for agent_key in sort_str_num(self.agent_keys):
            agent = self.agents[agent_key]
            last_actions = self._last_actions[agent_key]
            rot_x, rot_y = rad_rot_to_xy(agent.rot)
            agent_state = [sign*agent.pos[0] / self._pos_norm,
            sign*agent.pos[1] / self._pos_norm, sign*rot_x, sign*rot_y, last_actions[0], last_actions[1]]
            agent_state.extend(self.get_one_hot_id(agent_key))
            agent_state = np.array(agent_state, dtype=np.float32)
            if self.agents[agent_key].team == focus_agent.team:
                blue_agents.append(agent_state)
            else:
                red_agents.append(agent_state)

        blue_state = [time_ball_obs] + blue_agents + red_agents

        # Note: Red value function is not used in current training setup.
        # if self.players_per_team[1] > 0:
        #     assert self.players_per_team[0] == self.players_per_team[1]
        #     red_state = [copy.deepcopy(time_ball_obs)] + copy.deepcopy(red_agents) +\
        #                 copy.deepcopy(blue_agents)
        #
        #     # Player observations
        #     red_state[0][1:] *= -1
        #     for i in range(1, self.num_players+1):
        #         red_state[i][:-2] *= -1
        #
        #     # red_state[2:2 + self.players_per_team[0]] = -blue_state[2 + self.players_per_team[0]:]
        #     # red_state[2 + self.players_per_team[0]:] = -blue_state[2:2 + self.players_per_team[0]]
        #     # Flip all the actions back to what they were
        #     # red_state[1:][-2:] *= -1
        # else:
        #     red_state = None
        red_state = None

        return blue_state, red_state

    def _calc_rel_info(self, focus_pos, target_pos, obj_rot_view, focus_rot= None, x_rot=None, y_rot=None, ball=False, last_action=None):
        # Object is in vision range
        ff_obs = [1.0]
        obj_dist = dist(focus_pos, target_pos) / self._max_dist
        ff_obs.extend([obj_dist, obj_rot_view / (self.vision_range/2)])
        if x_rot is not None:
            rot = math.atan2(y_rot, x_rot)
            rel_x, rel_y = rad_rot_to_xy(rot-focus_rot)
            if ball:
                abs_val = dist((0, 0), (x_rot, y_rot))
                rel_x = rel_x*abs_val/max_ball_speed
                rel_y = rel_y*abs_val/max_ball_speed
            ff_obs.extend([rel_x, rel_y])

        if last_action is not None:
            ff_obs.extend([last_action[0], last_action[1]])
        return ff_obs
           
    def _get_rel_obs(self, focus_agent, target_object, last_action=None):
        cur_pos = focus_agent.pos
        circ_cen = target_object.pos
        obj_cen_rot = math.atan2(circ_cen[1] - cur_pos[1], circ_cen[0] - cur_pos[0])
        start_rot = (focus_agent.rot - self.vision_range / 2)
        end_rot = (focus_agent.rot + self.vision_range / 2)

        # Adjust rotations if necessary
        if start_rot > end_rot:
            start_rot -= 2 * math.pi
        if obj_cen_rot > end_rot:
            obj_cen_rot -= 2*math.pi
        elif obj_cen_rot < start_rot:
            obj_cen_rot += 2*math.pi

        if start_rot <= obj_cen_rot <= end_rot:
            obj_rot_view = obj_cen_rot - (start_rot+end_rot)/2
            if type(target_object) == Agent:
                rot_x, rot_y = rad_rot_to_xy(target_object.rot)
                ff_obs = self._calc_rel_info(focus_agent.pos, target_object.pos, focus_rot=focus_agent.rot,
                                                 x_rot=rot_x, y_rot=rot_y, obj_rot_view=obj_rot_view,
                                             last_action=last_action)
                # Surely we don't need ids for opponent players
                # ff_obs.extend(self.get_one_hot_id(target_object.name))
            elif type(target_object)==Ball:
                ff_obs = self._calc_rel_info(focus_agent.pos, target_object.pos, focus_rot=focus_agent.rot,
                                             x_rot=target_object.vel[0], y_rot=target_object.vel[1],
                                             obj_rot_view=obj_rot_view, ball=True)

            else:
                raise NotImplementedError("Unknown type: ", type(target_object))
        else:
            if type(target_object) == Agent:
                ff_obs = [0.0] * (7) # +11
            elif type(target_object) == Ball:
                ff_obs = [0.0] * 5
            else:
                raise NotImplementedError("Unknown type: ", type(target_object))
        return ff_obs
    
    def _get_abs_obs(self, agent, need_swap, last_action):
        # Add the time left, agent position, rotation, last_action
        rot_x, rot_y = rad_rot_to_xy(agent.rot)
        ff_obs = [1.0 - float(self.num_steps) / self.game_length,
                  agent.pos[0] / self._pos_norm,
                  agent.pos[1] / self._pos_norm,
                  rot_x, rot_y,
                  last_action[0], last_action[1]]

        if need_swap:
            # Invert the positions and velocities
            for i in range(1, len(ff_obs)-2):
                ff_obs[i] *= -1
        ff_obs.extend(self.get_one_hot_id(agent.name))
        return ff_obs

    def _calc_observations(self, goal_scored, done):
        # TODO: Maybe randomise the order of the agents in the states?
        observations = {}
        states = {}
        rewards = {}

        if self.game_setting in ["ppo_attention_state", "hybrid_attention_state",
                                "maddpg_attention_state"]:
            blue_state, red_state = self._calc_state_func()
      
        # Calculate the observations
        agent_obs = {agent_key: self._calc_obs_func(agent_key) for agent_key in self.agent_keys}

        for agent_key in self.agent_keys:
            if self.game_setting in ["ppo_xray_state", "ppo_attention_state"]:
                if self.agents[agent_key].team==self.agents["agent_0"].team:
                    state = blue_state
                else:
                    state = red_state
            elif self.game_setting in ["hybrid_attention_state", "maddpg_attention_state"]:
                if self.agents[agent_key].team==self.agents["agent_0"].team:
                    state = copy.copy(blue_state)
                    # TODO: Fix this. This is still risky as it assumes agents are numbered: agent_0 .. agent_21.
                    index = int(agent_key.split("_")[-1]) + 1
                    assert 0 < index <= self.players_per_team[0]
                    old_state = state[index]
                    state[index] = state[1]
                    state[1] = old_state
                else:
                    assert red_state == None
                    state = red_state 
            else:
                raise NotImplementedError("Not used..")
            observation = OLT(
                observation=agent_obs[agent_key],
                legal_actions={},
                terminal=np.asarray([done], dtype=np.float32),
            )
       
            observations[agent_key] = observation
            
            states[agent_key] = state
            assert goal_scored == False or goal_scored == "blue" or goal_scored == "red"
            if goal_scored:
                if goal_scored == self.agents[agent_key].team:
                    rewards[agent_key] = np.array(1.0, dtype=np.float32)
                else:
                    rewards[agent_key] = np.array(-1.0, dtype=np.float32)
            else:
                rewards[agent_key] = np.array(0.0, dtype=np.float32)
        
        # Used to determine if a goal was scored.
        self._home_goal_reward = 0.0
        if goal_scored:
            #if len(old_team_num) == 0:
            if goal_scored == self.agents["agent_0"].team:
                self._home_goal_reward = 1.0
            else:
                self._home_goal_reward = -1.0

        # Add optional touch foul penalty.
        if self._add_touch_penalty:
            penalty = 0.02
            min_dist = self.agents["agent_0"].radius * 2
            
            for t_i in range(2):
                for a_i in range(self.num_per_team):
                    for a_j in range(a_i+1, self.num_per_team):
                        offset = t_i * self.num_per_team
                        key_1, key_2 = self.agent_keys[a_i + offset], self.agent_keys[a_j + offset]
                        if dist(self.agents[key_1].pos, self.agents[key_2].pos) < min_dist:
                            rewards[key_1] -= penalty
                            rewards[key_2] -= penalty
        self.observations = observations
        self.states = states
        self.rewards = rewards

    def get_reward_shaping_info(self):
        reward_shaping_info = {}
        for agent_key in self.agent_keys:
            p_x, p_y = self.agents[agent_key].pos
            b_x, b_y = self.ball.pos
            rs_info = np.array([p_x/self._pos_norm, p_y/self._pos_norm, b_x/self._pos_norm,
                                b_y/self._pos_norm], dtype=np.float32)

            # Invert the coordinates if the agent scores to the left.
            if self.team_field_sides[self.agents[agent_key].team] == "right":
                rs_info = -rs_info

            reward_shaping_info[agent_key] = rs_info

        return reward_shaping_info

    def get_home_goal_reward(self):
        return self._home_goal_reward

    def _reset_field(self):
        # Set the intial starting noise
        if self.game_type == "dom_rand":
            # This last np.clip is just so that we can see the difficulty of the game after the 
            # max field size is reached.
            self.update_dom_rand_game_length()

        assert self.game_type != "dom_rand"

        agent_x_min = self.x_out_start
        agent_x_max = 0.0
        pos_noise = 0.05
        
        for a_i, agent_key in enumerate(self.agents.keys()):
            agent = self.agents[agent_key]
            # For domain randomisation switch sides
            place_side = self.team_field_sides[agent.team]

            if self._reset_setup == "position" and self.num_players == 22:             
                init_pos = init_position[a_i%11]
                new_x = np.clip((init_pos[1] + uniform(-pos_noise, pos_noise))*self.field_size[0], agent_x_min, agent_x_max)
                new_y = np.clip((init_pos[0] + uniform(-pos_noise, pos_noise))*self.field_size[0], self.y_out_start, self.y_out_end)

                agent.pos = [new_x, new_y]
            elif self._reset_setup in ["random", "position"]:
                agent.pos = [uniform(agent_x_min, agent_x_max), uniform(self.y_out_start, self.y_out_end)]
            else:
                raise NotImplementedError("Reset method not implemented: ", self._reset_setup)

            agent.rot = uniform(-1.0, 1.0)*math.pi

            if place_side == 'right':
                agent.pos[0] *= -1.0
                agent.pos[1] *= -1.0
                agent.rot += math.pi
                if agent.rot > 2*math.pi:
                    agent.rot -= 2*math.pi
 
        # print("Add this back in.")
        self.ball.vel = [0.0, 0.0]

        self.ball.pos = [uniform(self.x_out_start, self.x_out_end), uniform(self.y_out_start, self.y_out_end)]
        if self.team_field_sides["blue"] == "right":
            self.ball.pos[0] *= -1.0
        return None

    def draw_circle(self, field_surf, circle_col, head_col, coords, rot_x, rot_y, switch_coords=False, size=5):
        if switch_coords:
            coords = [-coords[0]*self._pos_norm,-coords[1]*self._pos_norm]
            rot_x = -rot_x
            rot_y = -rot_y
        else:
            coords = [coords[0]*self._pos_norm,coords[1]*self._pos_norm]

        offset = [self.field_size[0]/2, self.field_size[1]/2]
        pygame.draw.circle(field_surf, circle_col, [int((coords[0]+offset[0])*self.r_to_s), 
        int((coords[1]+offset[1])*self.r_to_s)], size)

        radius = dist((0,0), (rot_x, rot_y))*10
        rot = math.atan2(rot_y, rot_x)
        sin_comp = radius * math.sin(rot)
        cos_comp = radius * math.cos(rot)
        x3 = coords[0] + cos_comp + offset[0]
        y3 = coords[1] + sin_comp + offset[1]
        pygame.draw.line(field_surf, head_col, ((coords[0]+ offset[0]) * self.r_to_s, (coords[1]+ offset[1]) * self.r_to_s),
                        (x3 * self.r_to_s, y3 * self.r_to_s),
                        2)

    def _update_screen(self):
        # Plot field background
        field_surf = pygame.Surface(self.screen_size)
        field_surf.fill(field_col)

        # Draw play field
        pygame.draw.rect(field_surf, white, (
            self.x_out_screen_start, self.y_out_screen_start, self.in_field_size[0] * self.r_to_s,
            self.in_field_size[1] * self.r_to_s), 2)

        # Draw goals
        pygame.draw.rect(field_surf, white, (
            self.x_goal1_screen_start+2, self.y_goal_screen_start, self.goal_size[0] * self.r_to_s,
            self.goal_size[1] * self.r_to_s), 2)

        pygame.draw.rect(field_surf, white, (
            self.x_goal2_screen_start-1, self.y_goal_screen_start, self.goal_size[0] * self.r_to_s,
            self.goal_size[1] * self.r_to_s), 2)

        # Center line
        pygame.draw.line(field_surf, white, (
            self.cen_screen_x, self.cen_screen_y_1), (self.cen_screen_x, self.cen_screen_y_2-1), 2)

        # Draw agents
        for agent_key in self.agent_keys:
            agent = self.agents[agent_key]
            if agent_key != self.focus_player:
                agent.render(field_surf)
            elif self.show_agent_rays and self.observations is not None:
                agent.render(field_surf, field_size=self.field_size, vision_cone=self.observations[agent_key].observation[0],
                             state_view=self.states[agent_key])
            else:
                agent.render(field_surf)

        # Draw ball
        self.ball.render(field_surf)
        font = pygame.font.SysFont("comicsansms", 48)
        text = font.render(str(self.goals["blue"]) + " : " + str(self.goals["red"]), True, (0, 128, 0))
        field_surf.blit(text, (self.field_size[0] * self.r_to_s/2-32, 30))

        font = pygame.font.SysFont("comicsansms", 48)
        text = font.render("Step: " + str(self.num_steps), True, (0, 128, 0))
        field_surf.blit(text, (50, 30))

        self.screen.blit(field_surf, [0, 0])
        pygame.display.flip()

    def observation_spec(self) -> types.NestedSpec:
        return self._observation_specs

    def action_spec(self) -> types.NestedSpec:
        return self._action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
            reward_specs = {}
            for agent in self.agent_keys:
                reward_specs[agent] = specs.Array((), np.float32)
            return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        discount_specs = {}
        for agent in self.agents:
            discount_specs[agent] = specs.BoundedArray(
                (), np.float32, minimum=0, maximum=1.0
            )
        return discount_specs

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        return self._extra_specs

    def reset(self):
        self.num_steps = 0
        
        # Remove this assert if necessary
        assert self.dom_rand_use_all == False

        if self._game_diff < 1.0 and self.game_type == "dom_rand":
            score = self.goals["blue"] - self.goals["red"]
            result = 0.5
            if score < 0.0:
                result = 0.0
            elif score > 0.0:
                result = 1.0
            self._game_results.append(result)
            # import tensorflow as tf
            is_full = len(self._game_results)==self._game_results.maxlen
            if is_full and np.mean(self._game_results) > 0.75:
                self._game_diff += 0.05

                # Clear the queue
                self._game_results.clear()

            self._game_diff = np.clip(self._game_diff, 0.0, 1.0)
            self._field_diff = np.clip(self._game_diff, 0.0, 1.0)
            
        self.goals = {"blue": 0, "red": 0}
         # Switch team sides
        if self.do_team_switch:
            blue_side = self.team_field_sides['blue']
            self.team_field_sides['blue'] = self.team_field_sides['red']
            self.team_field_sides['red'] = blue_side

        self._reset_field()

        for agent_key in self.agent_keys:
            self._last_actions[agent_key] = np.zeros(2, dtype=np.float32)

        self._calc_observations(goal_scored=False, done=False)
        timestep = dm_env.restart(self.observations)

        return timestep, {"env_states": self.states}

    def step(self, actions):
        self.num_steps += 1
        for agent_key in self.agent_keys:
            action = actions[agent_key]
            if len(action.shape) == 0:
                # Convert discrete actions
                int_action = action
                action = np.zeros(2, dtype=np.float32)
                # Convert action
                action[self.a_dict["move"]] = int(int_action / 3)-1
                action[self.a_dict["rot"]] = (int_action % 3)-1

            self._last_actions[agent_key] = action
            agent = self.agents[agent_key]

            # Update rotation
            agent.rot -= action[self.a_dict["rot"]]*self.rot_speed
            if agent.rot > 2.0*math.pi:
                agent.rot -= 2.0 * math.pi
            elif agent.rot < 0.0:
                agent.rot += 2.0 * math.pi

            # Update movement
            assert -1.0 <= action[self.a_dict["move"]] <= 1.0
            assert -1.0 <= action[self.a_dict["rot"]] <= 1.0

            sin_comp = action[self.a_dict["move"]] * self.move_speed * math.sin(agent.rot)
            cos_comp = action[self.a_dict["move"]] * self.move_speed * math.cos(agent.rot)

            agent.pos[0] = agent.pos[0]+cos_comp
            agent.pos[1] = agent.pos[1]+sin_comp

            # This is much faster than np.clip :)
            agent.pos[0] = max(min(agent.pos[0]+cos_comp, self.x_out_end), self.x_out_start)
            agent.pos[1] = max(min(agent.pos[1]+sin_comp, self.y_out_end), self.y_out_start)

        if self._heatmap_save_loc:
            line = ""
            assert len(self.agents)==22
            for a_i in range(11):
                position = self.agents[f"agent_{a_i}"].pos
                line += f"{position[0]},{position[1]};"

            with open(self._heatmap_save_loc, 'a') as fp:
                fp.write(line + "\n")       

        # Add new contact velocity
        self.ball.check_hit(self.agents)

        # Update ball movement
        self.ball.pos[0] += self.ball.vel[0]
        self.ball.pos[1] += self.ball.vel[1]

        # Decrease ball speed due to friction and stop if to slow
        self.ball.vel[0] *= self.ball.fric
        self.ball.vel[1] *= self.ball.fric

        if speed(self.ball.vel) < 0.01:
            self.ball.vel[0] = 0.0
            self.ball.vel[1] = 0.0

        goal_scored = False
        if math.fabs(self.ball.pos[1]) <= 0.5*self.goal_size[1]:# Check if inline with goalpost
            if self.ball.pos[0] < self.x_out_start:  # Blue goal post
                goal_scored = 'red' if self.team_field_sides["red"] == "right" else "blue"
                self.goals[goal_scored] += 1
            elif self.ball.pos[0] > self.x_out_end:  # Red goal post
                goal_scored = 'blue' if self.team_field_sides["blue"] == "left" else "red"
                self.goals[goal_scored] += 1
        else:   # Check if ball is out above or below
            if self.ball.pos[0] < self.x_out_start:
                # self.ball.pos = pre_pos
                self.ball.pos[0] = self.x_out_start
                self.ball.vel[0] = -self.ball.vel[0]
                self.ball.vel = [self.ball.vel[0]*0.6, self.ball.vel[1]*0.6]
            elif self.ball.pos[0] > self.x_out_end:
                # self.ball.pos = pre_pos
                self.ball.pos[0] = self.x_out_end
                self.ball.vel[0] = -self.ball.vel[0]
                self.ball.vel = [self.ball.vel[0]*0.6, self.ball.vel[1]*0.6]
        
        # Check if ball is out
        if self.ball.pos[1] < self.y_out_start:
            # self.ball.pos = pre_pos
            self.ball.pos[1] = self.y_out_start
            self.ball.vel[1] = -self.ball.vel[1]
            self.ball.vel = [self.ball.vel[0]*0.6, self.ball.vel[1]*0.6]
        elif self.ball.pos[1] > self.y_out_end:
            # self.ball.pos = pre_pos
            self.ball.pos[1] = self.y_out_end
            self.ball.vel[1] = -self.ball.vel[1]
            self.ball.vel = [self.ball.vel[0]*0.6, self.ball.vel[1]*0.6]

        # Normal play
        if goal_scored:
            # Goal scored
            self._reset_field()
        
        done = self.game_length <= self.num_steps
        assert self.game_length >= self.num_steps

        self._calc_observations(goal_scored, done)
      
        if done:
            self._step_type = dm_env.StepType.LAST
        else:
            self._step_type = dm_env.StepType.MID

        if self.render_game:
            self._update_screen()
            if self.include_wait:
                time.sleep(self._wait_period)
        return (
            dm_env.TimeStep(
                observation=self.observations,
                reward=self.rewards,
                discount=self._discount,
                step_type=self._step_type,
            ),
            {"env_states": self.states},
        )
