import argparse
import os
import yaml
from env.simulator import *
from controller import *
from env.task import *
from utils import task_registry

ROBOT_CONTROL_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def main(args):
    config_filepath = os.path.join(ROBOT_CONTROL_ROOT_DIR,'config',f'{args.task}.yaml')
    with open(config_filepath, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    simulator= task_registry.get_components(cfg['simulator'], args.task, f"{args.task}_{cfg['controller']}", cfg)
    simulator.run_simulation()    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Simulation')
    root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
    parser.add_argument('--task', required=True, choices=['cartpole'])
    args = parser.parse_args()
    main(args)