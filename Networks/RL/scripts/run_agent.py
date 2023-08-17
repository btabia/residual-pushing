from test_agent_script import play
from train_agent_script import  train
import sys, os
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../push_gym/push_gym/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../push_gym/push_gym/tasks/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../push_gym/push_gym/environments/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../push_gym/push_gym/utils/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../push_gym/push_gym/utils/plot_scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../push_gym/push_gym/utils/Lazy-Theta-with-optimization-any-angle-pathfinding/build/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../push_gym/push_gym/envs/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../push_gym/push_gym/baseline_approach/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../push_gym/push_gym/evaluation/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../push_gym/push_gym/plot_scripts/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../VAE/models/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../push_gym/push_gym/rl_env/'))
#print(sys.path)
#if __name__ == "__main__":
#    params = []
#    file = open("parameters.yaml", 'r')
#    dict = yaml.load(file, Loader=yaml.FullLoader)
#    for key, value in dict.items():
#        print("Key: ", key, ", Value: ", value)
#        params.append(value)
#    if params[8]:
#        print("------ Training Model ------")
#        train_model(params[0], params[1], params[2], params[3], params[4], params[6], params[7], params[9], params[16],
#                    params[14], params[13], params[10], params[11], params[17], params[12])
#    else:
#        print("------ Testing Model ------")
#        test_agent(params[4], params[5], params[6], params[16], params[14], params[17], params[18], params[19])

@hydra.main(config_name="playing", config_path="./config")
def parse_hydra_configs(cfg: DictConfig) -> None:

    cwd = os.getcwd()
    hydra_folder_path = cwd + "/.hydra"
    config_file_path = hydra_folder_path + "/config.yaml"
    print("file path: " + str(cwd))

    if cfg["name"] == "training":
        print("------ Training Model ------")
        train(cfg)
    
    if cfg["name"] == "playing":
        print("------ Playing Model ------")
        play(cfg)


if __name__ == '__main__':
    parse_hydra_configs()