import argparse
import os
import yaml
from src.simulator import *
from src.controller import *
from src.task import *
from src.utils import registry
from src.utils.env import ROBOT_CONTROL_ROOT_DIR

def main(args):
    config_filepath = os.path.join(ROBOT_CONTROL_ROOT_DIR,'config',f'{args.task}',f'{args.task}.yaml')
    with open(config_filepath, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    simulator= registry.get_components(cfg['simulator'], args.task, f"{args.task}_{cfg['controller']}", cfg)
    simulator.run_simulation()   
    simulator.generate_report() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Simulation')
    root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
    parser.add_argument('--task', required=True, choices=['cartpole', 'flywheel', 'wheel_robot', 'vehicle'])
    args = parser.parse_args()
    main(args)