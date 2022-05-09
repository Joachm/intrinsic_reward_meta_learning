from my_gym.envs.mujoco.mujoco_env import MujocoEnv

# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from my_gym.envs.mujoco.ant import AntEnv
from my_gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from my_gym.envs.mujoco.hopper import HopperEnv
from my_gym.envs.mujoco.walker2d import Walker2dEnv
from my_gym.envs.mujoco.humanoid import HumanoidEnv
from my_gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from my_gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from my_gym.envs.mujoco.reacher import ReacherEnv
from my_gym.envs.mujoco.swimmer import SwimmerEnv
from my_gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from my_gym.envs.mujoco.pusher import PusherEnv
from my_gym.envs.mujoco.thrower import ThrowerEnv
from my_gym.envs.mujoco.striker import StrikerEnv
