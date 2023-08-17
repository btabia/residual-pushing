from stable_baselines3 import PPO
import gym, push_gym
import tensorflow as tf
from tqdm import tqdm
from stable_baselines3.common.env_util import make_vec_env
import json
import os
import torch as th
import shutil


def play(cfg) -> None:
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    #torch.device('cpu')
    th.set_num_threads(2)
    # Initialize environments
    print("------ Initialise Environment ------")
    env = gym.make(cfg["general"]["env_name"])
    #initialise environment configuration values
    env.setup_config(cfg)
    model = PPO.load(cfg["play"]["policy_path"])
    deterministic = cfg["play"]["deterministic"]

    exit = False
    iteration = 1
    base_dir = cfg["play"]["data_log_path"]
    saving_path = base_dir + "/play_" + str(iteration)
    while exit == False:
        folder_exist = os.path.exists(saving_path)
        if folder_exist == False: 
            os.makedirs(saving_path)
            exit = True
        elif folder_exist == True:
            iteration = iteration + 1
            saving_path = base_dir + "/play_" + str(iteration)
    

    shutil.copyfile("/home/btabia/git/RRL_pushing/cfg/multiparticles.yaml", saving_path + "/config.yaml")
    
    it = 0
    for _ in range(cfg["play"]["max_iteration"]):
        obs = env.reset()
        done = False
        cumul_reward = 0
        it = it + 1
        while not done: 
            actions, _ = model.predict(observation=obs, deterministic=deterministic)
            obs, reward, done, info = env.step(actions)
            print("instant reward: " + str(reward))
            cumul_reward = reward + cumul_reward
            print("cumulative reward: " + str(cumul_reward))
            #env.render()
        name_prefix = "/episode_" + str(it) + "_log.json"
        #env.save_logged_data(saving_path + name_prefix)
    env.close()


