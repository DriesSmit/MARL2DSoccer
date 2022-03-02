import numpy as np
from custom_football_env import dist

class NaiveTeamAttentionBot(object):
    def __init__(self, bot_diff=1.0):
        # self._agent_num = agent_num
        self._bot_diff = bot_diff

        self.is_attacking = None
        self.defend_y = None
        self.speed = None

    def observe_after(self, reward, action=None):
        pass

    def reset_brain(self):
        pass

    def get_action(self, all_observations, states=None, add_to_memory=False):
        observations = all_observations[0]
        num_actions = len(observations)

        if self.is_attacking is None:
            self.is_attacking = np.zeros(num_actions, dtype=np.int32)
            self._num_defending = int(num_actions/2)
            num_attacking = num_actions - self._num_defending
            assert self._num_defending > 3
            self.is_attacking[self._num_defending+1:] = 1
            self.defend_y = np.arange(-0.5, 0.5+1/self._num_defending, 1/self._num_defending)

            self.speed = np.arange(0.5, 1.0+1/num_attacking, 1/num_attacking)

        actions_list = []
        for a_i in range(num_actions):
            player_obs = observations[a_i]
            agent_x, agent_y, deg_x, deg_y = player_obs[1:5]

            see_ball, ball_dist, ball_dir = player_obs[-5:-2]

            a_dict = {"move": 0, "rot": 1}
            actions = np.zeros(2, dtype=np.float32)

            # Defenders go to position
            if not self.is_attacking[a_i]:
                defend_pos = (-0.9, self.defend_y[a_i])
                if a_i == 0 or a_i == len(self.defend_y)-1:
                    defend_pos = (-1.0, self.defend_y[a_i])

                dist_to_cen = dist(defend_pos, (agent_x, agent_y))
                if see_ball > 0.5 and ball_dist < 0.1 and agent_x < 0.0:  # Defenders attacking ball

                    # Middle defenders kick to the side of the field
                    if agent_y < 0.0:
                        targ_rot_y = -1.0 - agent_y
                        targ_rot_x = -0.7 - agent_x
                    else:
                        targ_rot_y = 1.0 - agent_y
                        targ_rot_x = -0.7 - agent_x
                    if a_i == 0 or a_i == len(self.defend_y) - 1: # Side defenders kick forward
                        targ_rot_y = 0 - agent_y
                        targ_rot_x = 1 - agent_x
                    target_rot = np.arctan2(targ_rot_y, targ_rot_x)
                    agent_rot = np.arctan2(deg_y, deg_x)
                    deg_diff = target_rot - agent_rot
                    actions[a_dict["move"]] = 1.0
                elif dist_to_cen < 0.1: # Defenders waiting
                    if see_ball > 0.5:
                        deg_diff = ball_dir
                    else:
                        targ_rot_y = 1 - agent_y
                        targ_rot_x = 0 - agent_x
                        target_rot = np.arctan2(targ_rot_y, targ_rot_x)
                        agent_rot = np.arctan2(deg_y, deg_x)
                        deg_diff = target_rot - agent_rot
                    actions[a_dict["move"]] = 0.0
                else: # Defenders go to position
                    targ_rot_y = defend_pos[1] - agent_y
                    targ_rot_x = defend_pos[0] - agent_x
                    target_rot = np.arctan2(targ_rot_y, targ_rot_x)
                    agent_rot = np.arctan2(deg_y, deg_x)
                    deg_diff = np.pi + (target_rot - agent_rot)
                    actions[a_dict["move"]] = -1.0
            else: # Attackers
                if see_ball > 0.5:
                    deg_diff = ball_dir
                    actions[a_dict["move"]] = 1.0 # self.speed[a_i-self._num_defending-1]

                else: # Move backwards until attacker see the ball
                    actions[a_dict["move"]] = -1.0
                    targ_rot_y = 0 - agent_y
                    targ_rot_x = 1 - agent_x
                    target_rot = np.arctan2(targ_rot_y, targ_rot_x)
                    agent_rot = np.arctan2(deg_y, deg_x)
                    deg_diff = target_rot - agent_rot

                # Kick ball if is close to the ball
                if see_ball > 0.5 and ball_dist < 0.04:
                    actions[a_dict["move"]] = 1.0
                    targ_rot_y = 0 - agent_y
                    targ_rot_x = 1 - agent_x
                    target_rot = np.arctan2(targ_rot_y, targ_rot_x)
                    agent_rot = np.arctan2(deg_y, deg_x)
                    deg_diff = target_rot - agent_rot

            if deg_diff > np.pi:
                deg_diff -= 2 * np.pi
            elif deg_diff < -np.pi:
                deg_diff += 2 * np.pi

            rot_speed = -deg_diff

            if rot_speed > 1:
                rot_speed = 1
            elif rot_speed < -1:
                rot_speed = -1

            actions[a_dict["rot"]] = rot_speed  # np.random.uniform(-1.0, 1.0)

            if np.random.uniform(0, 1) < 1.0-self._bot_diff:
                actions[a_dict["move"]] = np.random.uniform(-1, 1)
                actions[a_dict["rot"]] = np.random.uniform(-1.0, 1.0)

            # Clip the actions if they are out of range.
            actions = np.clip(actions, -1.0, 1.0)

            actions_list.append(actions)
        return actions_list