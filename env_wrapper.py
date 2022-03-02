import tensorflow as tf
from typing import Any, NamedTuple
from custom_football_env import FootballEnv as CustomFootballEnv
import dm_env
import numpy as np
from sort_utils import sort_str_num
class EnvironmentSpec(NamedTuple):
    observation_spec: Any
    state_spec: Any
    action_spec: Any
    reward_spec: Any
    action_log_prob_spec: Any

class FootballEnvWrapper:
    """Environment wrapper for 2D Football environment."""

    def __init__(self, render, num_per_team, do_team_switch=False, env_obs_type = "ppo_attention", action_space = "continuous",
                 game_step_lim=400, show_env_feedback=False, include_wait=False, heatmap_save_loc=None,
                 reset_setup="position"):
        self._num_per_team = num_per_team
       
        self._action_space = action_space
        game_setting = f"{env_obs_type}_state"

        self._environment = CustomFootballEnv(render_game=render, game_setting=game_setting,
                                    players_per_team=[num_per_team, num_per_team],
                                    do_team_switch=do_team_switch, include_wait=include_wait,
                                    game_length=game_step_lim,
                                    game_diff=1.0,
                                    vision_range=np.pi, show_agent_rays=show_env_feedback,
                                    heatmap_save_loc=heatmap_save_loc, reset_setup=reset_setup)
        self.num_channels = CustomFootballEnv.get_num_channels()
        self.prefix = "agent"

        # Don't change the game length again.
        self._environment.game_type = "fixed"

    def update_screen(self):
        self._environment._update_screen()

    def get_game_length(self):
        return self._environment.game_length

    def reset_game(self):
        timestep, extras = self._environment.reset()
        batch_timestep = self.convert_and_batch_step(timestep, extras)
        return batch_timestep

    def step(self, actions):
        dict_actions = {}
        for a_i, agent_key in enumerate(self._environment.agent_keys):
            dict_actions[agent_key] = actions[a_i]
        timestep, extras = self._environment.step(dict_actions)
        done = timestep.step_type == dm_env.StepType.LAST
        observations, states, rewards = self.convert_and_batch_step(timestep, extras)
        return observations, states, rewards, done

    def convert_and_batch_step(self, timestep, extras):
        obs_list = [[] for _ in range(len(timestep.observation[f"{self.prefix}_0"].observation))]
        state_list = [[] for _ in range(len(extras["env_states"][f"{self.prefix}_0"]))]
        rewards = []

        for agent in sort_str_num(self._environment.agent_keys):
            obs = timestep.observation[agent].observation
            for i in range(len(obs_list)):
                obs_list[i].append(obs[i])
            state = extras["env_states"][agent]
            if state is not None:
                for i in range(len(state_list)):
                    state_list[i].append(state[i])
            rewards.append(timestep.reward[agent] if timestep.reward is not None else 0.0)
        # Batch the observations
        obs_team = [np.stack(obs_list[i][:self._num_per_team]) for i in range(len(obs_list))]
        obs_opp = [np.stack(obs_list[i][self._num_per_team:]) for i in range(len(obs_list))]
        observations = [obs_team, obs_opp]

        assert len(state_list[0]) == self._num_per_team
        state_team = [np.stack(state_list[i][:self._num_per_team]) for i in range(len(state_list))]
        states = [state_team, None] #state_opp]

        rewards = [rewards[:self._num_per_team], rewards[self._num_per_team:]]
        return observations, states, rewards

    def get_specs(self):
        OBSERVATION_SPEC = self._environment.observation_spec()[f"{self.prefix}_0"].observation
        STATE_SPEC = self._environment.extra_spec()["env_states"][f"{self.prefix}_0"]
        REWARD_SPEC = self._environment.reward_spec()[f"{self.prefix}_0"]
        ACTION_LOG_PROB_SPEC = tf.TensorSpec([], tf.float32)

        if self._action_space=="discrete":
            ACTION_SPEC = tf.TensorSpec([], tf.int32)
            raise NotImplementedError("This has not been used in a long time.")
        else:
            ACTION_SPEC = self._environment.action_spec()[f"{self.prefix}_0"]
        return EnvironmentSpec(observation_spec=[tf.TensorSpec.from_spec(OBSERVATION_SPEC[i]) for i in range(len(OBSERVATION_SPEC))],
                                state_spec=[tf.TensorSpec.from_spec(STATE_SPEC[i]) for i in range(len(STATE_SPEC))],
                                action_spec=tf.TensorSpec.from_spec(ACTION_SPEC),
                                reward_spec=tf.TensorSpec.from_spec(REWARD_SPEC),
                                action_log_prob_spec=tf.TensorSpec.from_spec(ACTION_LOG_PROB_SPEC))
