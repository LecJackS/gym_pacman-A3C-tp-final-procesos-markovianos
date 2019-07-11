"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import create_train_env
from src.model import Mnih2016ActorCritic
AC_NN_MODEL = Mnih2016ActorCritic

from src.optimizer import GlobalRMSProp
from src.process import local_train, local_test
import torch.multiprocessing as _mp
import shutil


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for Super Mario Bros""")
    parser.add_argument("--layout", type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument("--num_local_steps", type=int, default=50)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=2)
    parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/a3c_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_previous_weights", type=bool, default=True,
                        help="Load weight from previous trained stage")
    parser.add_argument("--use_gpu", type=bool, default=True)
    args = parser.parse_args()
    return args


def train(opt):
    torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    mp = _mp.get_context("spawn")
    env, num_states, num_actions = create_train_env(opt.layout)
    #global_model = ActorCritic(num_states, num_actions)
    global_model = AC_NN_MODEL(num_states, num_actions)
    if opt.use_gpu:
        global_model.cuda()
    global_model.share_memory()
    if opt.load_previous_weights:
#         if opt.stage == 1:
#             previous_world = opt.world - 1
#             previous_stage = 4
#         else:
#             previous_world = opt.world
#             previous_stage = opt.stage - 1
        file_ = "{}/gym-pacman_{}".format(opt.saved_path, opt.layout)
        if os.path.isfile(file_):
            print("Loading previous weights for %s..." %opt.layout, end=" ")
            global_model.load_state_dict(torch.load(file_))
            print("Done.")
        else:
            print("Can't load any previous weights for %s!" %opt.layout)
#             print("Loading some other map...", end=" ")
#             first_layout = "microGrid_superEasy1"
#             file_ = "{}/gym-pacman_{}".format(opt.saved_path, first_layout)
#             if os.path.isfile(file_):
#                 global_model.load_state_dict(torch.load(file_))
#                 print("Done.")
#             else:
#                 print("Failed.")
    #optimizer = GlobalAdam(global_model.parameters(), lr=opt.lr)
    optimizer = GlobalRMSProp(global_model.parameters(), lr=opt.lr)
    processes = []
    for index in range(opt.num_processes):
        # Multiprocessing async agents
        if index == 0:
            process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer, True))
        else:
            process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer))
        process.start()
        processes.append(process)
    # Local test simulation
    #process = mp.Process(target=local_test, args=(opt.num_processes, opt, global_model))
    #process.start()
    #processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
