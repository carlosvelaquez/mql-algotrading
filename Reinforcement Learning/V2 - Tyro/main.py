from rl_coach.agents.rainbow_dqn_agent import RainbowDQNAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment, GymEnvironmentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import LinearSchedule

TICKS_PER_MINUTE = 6
# HISTORY_MINUTES = 100
# EPISODE_MINUTES = int(60*24)
HISTORY_MINUTES = 10
EPISODE_MINUTES = 60

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(100000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10)
schedule_params.evaluation_steps = EnvironmentEpisodes(5)
schedule_params.heatup_steps = EnvironmentSteps(100)

#########
# Agent #
#########
agent_params = RainbowDQNAgentParameters()

# DQN params
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(
    100)
agent_params.algorithm.discount = 0.99
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)

# NN configuration
agent_params.network_wrappers['main'].learning_rate = 0.00025
agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False

# ER size
agent_params.memory.max_size = (MemoryGranularity.Transitions, 100000)

# agent_params.memory.beta = LinearSchedule(0.4, 1, 10000)
# agent_params.memory.alpha = 0.5

################
#  Environment #
################
env_params = GymVectorEnvironment(level='./tyro.py:Tyro')
#env_params.observation_space_type = BoxActionSpace
env_params.additional_simulator_parameters = {
    "history_file_path": "history.csv",
    "history_timesteps": int(TICKS_PER_MINUTE*HISTORY_MINUTES),
    "ticks_per_episode": int(TICKS_PER_MINUTE*EPISODE_MINUTES),
    "debug_mode": 0,
    "risk": 0.01,
    "fixed_position_size": None
}

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True
preset_validation_params.min_reward_threshold = EPISODE_MINUTES
preset_validation_params.max_episodes_to_achieve_reward = 250

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params, name="Tyro")
