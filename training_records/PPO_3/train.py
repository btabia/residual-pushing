from stable_baselines3 import TD3, PPO, SAC
from stable_baselines3.ppo import MlpPolicy
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from custom_callbacks import SavingCallback, ProgressBarManager, CurriculumCallback
import os
import gym, push_gym
import numpy as np
import torch as th
import tensorflow as tf
from stable_baselines3.common.env_util import make_vec_env
from policy_network import CNN
from typing import Callable
from push_gym.tasks.pushing import Pushing
import hydra
from omegaconf import DictConfig, OmegaConf
import shutil
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, env, verbose = 0):
      super().__init__(verbose)
      self.env = env
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> bool:

        logging_data = self.env.get_data_training_log(30000)
        self.mean_total_reward = logging_data[0]
        self.mean_disp_reward = logging_data[1]
        self.mean_dist_reward = logging_data[2]

        self.mean_ee_object_speed =  logging_data[3]
        self.mean_ee_object_dist = logging_data[4]

        self.mean_object_target_speed  = logging_data[5]
        self.mean_object_target_dist = logging_data[6]

        self.mean_normal_total_force = logging_data[7]
        self.mean_normal_residual_force  = logging_data[8]
        self.mean_normal_impedance_force = logging_data[9]

        self.logger.record("rewards/total_reward", self.mean_total_reward)
        self.logger.record("rewards/displacement_reward", self.mean_disp_reward)
        self.logger.record("rewards/distance_reward", self.mean_dist_reward)

        self.logger.record("observations/ee_object_speed", self.mean_ee_object_speed)
        self.logger.record("observations/ee_object_distance", self.mean_ee_object_dist)

        self.logger.record("observations/object_target_speed", self.mean_object_target_speed)
        self.logger.record("observations/object_target_distance", self.mean_object_target_dist)

        self.logger.record("actions/total_force", self.mean_normal_total_force)
        self.logger.record("actions/interaction_force", self.mean_normal_impedance_force)
        self.logger.record("actions/residual_force", self.mean_normal_residual_force)

        self.env.reset_logging_data()
        return True

    
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def setup_PPO(cfg, env, saving_path):
    policy = "MlpPolicy"
    activation = th.nn.ReLU
    vf_nn = OmegaConf.to_container(cfg["RL"]["policy_kwargs"]["nn_config"]["value_function"])
    policy_nn = OmegaConf.to_container(cfg["RL"]["policy_kwargs"]["nn_config"]["policy"])
    policy_kwargs = dict(activation_fn = activation, net_arch=(dict(vf= vf_nn, pi= policy_nn)))
    policy = MlpPolicy
    model = PPO(
                policy, 
                env, 
                policy_kwargs=policy_kwargs,
                verbose = cfg["RL"]["PPO"]["verbose"],
                n_steps = cfg["RL"]["PPO"]["n_steps"],
                batch_size = cfg["RL"]["PPO"]["batch_size"],
                learning_rate = linear_schedule(cfg["RL"]["PPO"]["learning_rate"]),
                gamma = cfg["RL"]["PPO"]["gamma"], 
                ent_coef = cfg["RL"]["PPO"]["ent_coef"],
                clip_range = cfg["RL"]["PPO"]["clip_range"], 
                n_epochs = cfg["RL"]["PPO"]["n_epochs"],
                gae_lambda = cfg["RL"]["PPO"]["gae_lambda"], 
                max_grad_norm = cfg["RL"]["PPO"]["max_grad_norm"],
                vf_coef = cfg["RL"]["PPO"]["vf_coef"],
                device = cfg["RL"]["PPO"]["device"],
                tensorboard_log = saving_path,
            )
    if cfg["RL"]["load_policy"] == True:
        model.set_parameters(cfg["RL"]["load_policy_path"])

    return model


def train_model(env_file_name, train_file_name, obj_creation_file_name, curric_task_file_name, tf_model_path,
                log_dir_name, tensorboard_log_name, load_model, obstacle_num, env_name, max_timesteps, curriculum,
                algorithm, arm_position, policy):
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    #torch.device('cpu')
    th.set_num_threads(2)

    os.makedirs(log_dir_name, exist_ok=True)

    # Initialize environments
    print("------ Initialise Environment ------")
    env = gym.make(env_name)
    #env.setup_for_RL(tf_model, obstacle_num, arm_position, curriculum, with_evaluation=False)

    checkpoint_path = "/home/btabia/git/residual-pushing/Networks/RL/tensorboard_logs"

    checkpoint_callback = CheckpointCallback(save_freq = 200000, save_path = checkpoint_path + "/checkpoints", name_prefix = "checkpoint")
    tensorboard_callback = TensorboardCallback(env)
    callback_list = CallbackList([checkpoint_callback, tensorboard_callback])

    print("------ Loading Model ------")
    print("------ PPO ------")
    model = get_PPO(env=env)

    print("------  Model Learning ------")
    model.learn(total_timesteps=max_timesteps, callback = callback_list)

    print("------  Saving Model Policy ------")
    model.save(log_dir_name + "policy")
    env._p.unloadPlugin(env.plugin)
    print("------  Closing Environment ------")
    env.close()
   # eval_env.close()


def train(cfg):
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    #torch.device('cpu')
    th.set_num_threads(2)

    os.makedirs(cfg["general"]["log_dir_name"], exist_ok=True)

    # Initialize environments
    print("------ Initialise Environment ------")
    env = gym.make(cfg["general"]["env_name"])
    #initialise environment configuration values
    env.setup_config(cfg)

    checkpoint_path = "/home/btabia/git/residual-pushing/Networks/RL/tensorboard_logs"
    saving_freq = cfg["RL"]["checkpoint"]["saving_freq"]
    base_dir = cfg["RL"]["checkpoint"]["base_directory"]
    exit = False
    iteration = 1
    saving_path = base_dir + "/PPO_" + str(iteration)
    while exit == False:
        folder_exist = os.path.exists(saving_path)
        if folder_exist == False: 
            os.makedirs(saving_path)
            exit = True
        elif folder_exist == True:
            iteration = iteration + 1
            saving_path = base_dir + "/PPO_" + str(iteration)
    name_prefix = "checkpoint"

    checkpoint_callback = CheckpointCallback(save_freq = saving_freq, save_path = saving_path + "/checkpoints", name_prefix = "checkpoint")
    tensorboard_callback = TensorboardCallback(env)
    callback_list = CallbackList([checkpoint_callback, tensorboard_callback])

    print("log and policy saving path:" + saving_path)

    print("------ Saving Env Files ------")
    current_path = os.getcwd()
    print("current path: " + str(current_path))
    task_config = "/home/btabia/git/residual-pushing/Networks/RL/scripts/config/training.yaml"
    task_source = "/home/btabia/git/residual-pushing/push_gym/push_gym/tasks/pushing.py"
    train_source = "/home/btabia/git/residual-pushing/Networks/RL/scripts/train_agent_script.py"
    ur_env_source = "/home/btabia/git/residual-pushing/push_gym/push_gym/environments/ur5_environment.py"
    general_env_source = "/home/btabia/git/residual-pushing/push_gym/push_gym/environments/base_environment.py"

    shutil.copyfile(task_config, saving_path + "/training.yaml")
    shutil.copyfile(task_source, saving_path + "/pushing.py")
    shutil.copyfile(train_source, saving_path + "/train.py")
    shutil.copyfile(ur_env_source, saving_path + "/ur5_environment.py")
    shutil.copyfile(general_env_source, saving_path + "/base_environment.py")

    print("------ Loading Model ------")
    print("------ PPO ------")
    model = setup_PPO(cfg = cfg, env=env, saving_path=saving_path)

    print("------  Model Learning ------")
    model.learn(total_timesteps=cfg["RL"]["max_timestep"], callback = callback_list)

    print("------  Saving Model Policy ------")
    model.save(cfg["RL"]["log_dir_name"] + "policy")
    env._p.unloadPlugin(env.plugin)
    print("------  Closing Environment ------")
    env.close()
