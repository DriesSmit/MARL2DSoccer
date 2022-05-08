from env_wrapper import FootballEnvWrapper
from fixed_agent import NaiveTeamAttentionBot

# Initialise the football environment
football_env = FootballEnvWrapper(num_per_team=11, render=True, include_wait=True)

while True:
    # Start new game
    teams = [NaiveTeamAttentionBot(), NaiveTeamAttentionBot()]
    for i in range(len(teams)):
        teams[i].reset_brain()
    observations, states, rewards = football_env.reset_game()
    
    done = False
    while not done:

        # Get the agent actions
        actions = []
        for a_i in range(len(teams)):
            actions.extend(teams[a_i].get_action(observations[a_i], states[a_i], add_to_memory=False))

        # Update the football screen
        football_env.update_screen()

        # Environment steps
        observations, _, rewards, done = football_env.step(actions)
