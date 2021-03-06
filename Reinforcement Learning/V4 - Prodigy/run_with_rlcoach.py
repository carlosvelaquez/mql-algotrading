from rl_coach.agents.rainbow_dqn_agent import RainbowDQNAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment, GymEnvironmentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import LinearSchedule
from rl_coach.spaces import PlanarMapsObservationSpace, ImageObservationSpace
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter, InputFilter
from rl_coach.filters.observation.observation_rescale_to_size_filter import ObservationRescaleToSizeFilter

import numpy as np
####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(1000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentEpisodes(1)

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
agent_params.network_wrappers['main'].learning_rate = 0.0000625
agent_params.network_wrappers['main'].optimizer_epsilon = 1.5e-4
agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False

# ER size
agent_params.memory.max_size = (MemoryGranularity.Transitions, 100000)

################
#  Environment #
################
env_params = GymEnvironmentParameters(level='./prodigy.py:Prodigy')
env_params.default_input_filter = NoInputFilter()
env_params.default_output_filter = NoOutputFilter()

env_params.additional_simulator_parameters = {
    "mt5_path": r'C:\Users\imado\AppData\Roaming\MetaTrader 5 - EXNESS\terminal64.exe',
    "debug_mode": 2,
}

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True
preset_validation_params.min_reward_threshold = 250
preset_validation_params.max_episodes_to_achieve_reward = 250

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params, name="Prodigy")
