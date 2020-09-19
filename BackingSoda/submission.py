#  Copyright (C) 2020 Le Pham Tuyen. All rights reserved.
#  This code is only intended for the competition hosted by Dacon and LG "https://dacon.io/competitions/official/235612/overview/"
#  Anyone who wishes to use this code for a commercial use should contact the developer listed below.
# 
#  Developer: tuyenple
#  Developer's email: tuyenkhu@gmail.com

## 1. Library
# Importing appropriate libraries
import argparse
import datetime
import scipy.signal
import random
import torch
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import namedtuple
from torch.distributions import Categorical
import torch.nn as nn
from torch.optim import Adam
import logging
import itertools
from joblib import Parallel, delayed, cpu_count
from copy import deepcopy

logger = logging.getLogger('Toy')

EnvInfo = namedtuple("EnvInfo", ['current_time_idx',
                                 'event_a',
                                 'mol_a',
                                 'event_b',
                                 'mol_b',
                                 'part_1',
                                 'part_2',
                                 'part_3',
                                 'part_4'])

# 28시간 검사를 완료했는지 검사, CHECK Event시 -1, processtime_time >=98 이면 28
CHECK_TIME = 28

# 생산 가능 여부, 0 이면 28 시간 검사 필요
PROCESS_YES = 1
PROCESS_NO = 0

# 생산 물품 번호 1~4, stop시 0
CHECK_MODE_1 = 0
CHECK_MODE_2 = 1
CHECK_MODE_3 = 2
CHECK_MODE_4 = 3

# 생산시간이 얼마나 지속되었는지 검사, PROCESS +1, CHANGE +1, 최대 140
MIN_PROCESS_TIME = 98
MAX_PROCESS_TIME = 140

# Event ID
EVENT_CHECK_1 = 0
EVENT_CHECK_2 = 1
EVENT_CHECK_3 = 2
EVENT_CHECK_4 = 3
EVENT_PROCESS = 4

# Defining Variables
PART_RATE = 0.985
MOLDING_RATE = 0.975
MAX_CUT = {'BLK_1': 506, 'BLK_2': 506, 'BLK_3': 400, 'BLK_4': 400}
BLK_TO_MOL = {'BLK_1': 'MOL_1', 'BLK_2': 'MOL_2', 'BLK_3': 'MOL_3', 'BLK_4': 'MOL_4'}

# 1. Data
# Reading data from .csv files
stock_df = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'data/stock.csv'))
max_count_df = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'data/max_count.csv'))
submission_df = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'data/sample_submission.csv'))
submission_df['time'] = pd.to_datetime(submission_df['time'])
order_df = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'data/order.csv'))
for i in range(91):
    order_df.loc[91 + i, :] = ['2020-07-01', 0, 0, 0, 0]
order_df['time'] = pd.to_datetime(order_df['time']) + pd.Timedelta(hours=18)


# Defining cut rates for they differ by month
def get_cut_rate(blk, month):
    if blk == 'BLK_1':
        return 0.851
    elif blk == 'BLK_2':
        return 0.901
    elif blk == 'BLK_3':
        if month == 4:
            return 0.710
        elif month == 5:
            return 0.742
        elif month == 6:
            return 0.759
        else:
            return 0.0
    elif blk == 'BLK_4':
        if month == 4:
            return 0.700
        elif month == 5:
            return 0.732
        elif month == 6:
            return 0.749
        else:
            return 0.0


# Define Check and Process names. We only dealt with "Check" and "Process" events
def get_event_name_from_id(event_id):
    if event_id == EVENT_CHECK_1:
        return 'CHECK_1'
    elif event_id == EVENT_CHECK_2:
        return 'CHECK_2'
    elif event_id == EVENT_CHECK_3:
        return 'CHECK_3'
    elif event_id == EVENT_CHECK_4:
        return 'CHECK_4'
    elif event_id == EVENT_PROCESS:
        return 'PROCESS'


# calculated from maxcount.csv
def get_max_process(check_mode, month):
    if check_mode == CHECK_MODE_2:  # For better convergence of blk2
        return 1.0
    else:
        if month == 4:
            return 5.8579
        elif month == 5:
            return 5.8668
        elif month == 6:
            return 5.8757


def discount_cumsum(x, discount):
    """
    magic function to calculate discounted reward without reverse a list
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


## Data Processing
class ToySubmitter:

    def __init__(self, log_path='submissions'):
        self._log_path = log_path
        self._log_filename = f"train-{datetime.datetime.now().isoformat()}.log"
        self.submission = pd.read_csv(os.path.join(Path(__file__).resolve().parent,
                                                   'data/sample_submission.csv'))

        self.reset()
        self.best_reward = np.zeros(5)
        self.best_dacon_score = 0.8  # baseline score

    def reset(self):
        self.submission.loc[:, 'PRT_1':'PRT_4'] = 0

    def step(self, env_info):

        self.submission.loc[env_info.current_time_idx, 'Event_A'] = get_event_name_from_id(env_info.event_a)
        self.submission.loc[env_info.current_time_idx, 'MOL_A'] = env_info.mol_a
        self.submission.loc[env_info.current_time_idx, 'Event_B'] = get_event_name_from_id(env_info.event_b)
        self.submission.loc[env_info.current_time_idx, 'MOL_B'] = env_info.mol_b
        self.submission.loc[env_info.current_time_idx, 'PRT_1'] = env_info.part_1
        self.submission.loc[env_info.current_time_idx, 'PRT_2'] = env_info.part_2
        self.submission.loc[env_info.current_time_idx, 'PRT_3'] = env_info.part_3
        self.submission.loc[env_info.current_time_idx, 'PRT_4'] = env_info.part_4

    # When terminating, the data is now inputted into the .csv file for submission
    def terminate(self, dacon_score, check_reward, process_rewards, train_mode, epoch, itr):
        if not os.path.exists(self._log_path):
            os.makedirs(self._log_path)
        # Deciding which score to use based on the train mode
        if train_mode == -1:
            self.evaluate_reward = dacon_score
        else:
            self.evaluate_reward = check_reward if train_mode == 0 else process_rewards[train_mode - 1]

        # Submission
        time_text = f"{epoch}-{itr}-{datetime.datetime.now().isoformat()}"
        if self.best_reward[train_mode] < self.evaluate_reward:
            # Calculating Part numbers after calculating mol numbers
            PRTs = self.submission[['PRT_1', 'PRT_2', 'PRT_3', 'PRT_4']].values
            PRTs = (PRTs[:-1] - PRTs[1:])[24 * 23:]
            PRTs = np.ceil(PRTs * 1.1) + 1
            PAD = np.zeros((24 * 23 + 1, 4))
            PRTs = np.append(PRTs, PAD, axis=0).astype(int)

            # Saving
            self.submission.loc[:, 'PRT_1':'PRT_4'] = PRTs

            # Update best score for each model
            self.best_reward[0] = check_reward
            self.best_reward[1:] = process_rewards[:]

        # Save the csv file with best dacon score
        if self.best_dacon_score < dacon_score:
            csv_filename = self._log_path + f"/{time_text}.csv"
            self.submission.to_csv(csv_filename, index=False)
            self.best_dacon_score = dacon_score

        # Printing out each reward in terminal and saving in a .log file 
        text_out = f"Epoch: {epoch} | " \
                   f"Episode: {itr} | " \
                   f"Dacon Score: {dacon_score} | " \
                   f"Check Reward: {check_reward} | " \
                   f"BLK_1 Reward {process_rewards[0]} | " \
                   f"BLK_2 Reward {process_rewards[1]} | " \
                   f"BLK_3 Reward {process_rewards[2]} | " \
                   f"BLK_4 Reward {process_rewards[3]}"

        self.log(text_out)
        self.reset()

    # Destination of the log file
    def log(self, text, to_console=True, to_file=True):
        if to_console:
            print(text)

        if to_file:
            log_filename = self._log_path + f'/{self._log_filename}'
            file = open(log_filename, 'a')
            file.write(f"{text}\n")
            file.close()


## 4. Feature Engineering & Initial Modeling
# Creating an environment for performing RL
# We completely convert from Simulator (given by Organizer) to ToyEnv (similar to GymAI environment)
class ToyEnv:

    def __init__(self, check_offset_days=30, process_offset_days=30):

        self.check_offset_days = check_offset_days
        self.process_offset_days = process_offset_days

        self.process_actions = [0, 1]
        # Check actions
        check_params = {
            "check_a": [EVENT_CHECK_1, EVENT_CHECK_2, EVENT_CHECK_3, EVENT_CHECK_4],
            "check_b": [EVENT_CHECK_1, EVENT_CHECK_2, EVENT_CHECK_3, EVENT_CHECK_4]
        }
        # 16 discrete action (4x4) for check_model
        self.check_actions = list(dict(zip(check_params.keys(), values)) for values in
                                  itertools.product(*check_params.values()))

        # Define state dimension which depends on number of offset days
        self.check_obs_dim = int((self.check_offset_days + 1) * 4 + stock_df.shape[1])
        self.process_obs_dim = int((self.process_offset_days + 1) * 1 + stock_df.shape[1] // 4 + 1)
        self.reset()

    # Setting the initial check event name
    def set_initial_check_mode(self, initial_check_a=CHECK_MODE_1, initial_check_b=CHECK_MODE_1):
        self.check_mode[0] = initial_check_a
        self.check_mode[1] = initial_check_b

    # Resetting the environment
    def reset(self):
        self.current_time_idx = 0
        self.current_order_idx = 0

        self.check_time = [CHECK_TIME, CHECK_TIME]
        self.process = [PROCESS_NO, PROCESS_NO]

        self.check_mode = [CHECK_MODE_1, CHECK_MODE_1]

        self.process_time = [0, 0]
        self.running_mols = [[], []]
        self.inqueue_mols = np.zeros([2, 4])
        self.current_event_ids = [EVENT_CHECK_1, EVENT_CHECK_1]
        # 블럭 장난감 총 수요
        self.N = np.array([0, 0, 0, 0])
        # 수요 발생 시 블럭 장난감 생산 부족분 합계
        self.p = np.array([0, 0, 0, 0])
        # 수요 발생 시 블럭 장난감 생산 초과분 합계
        self.q = np.array([0, 0, 0, 0])

        self.running_stock = submission_df.copy()
        self.running_stock.drop(self.running_stock.columns, axis=1, inplace=True)

        for col in stock_df.columns.tolist():
            self.running_stock[col] = 0.0
            self.running_stock.loc[0, col] = stock_df.loc[0, col]

        return self._get_obs()

    def change_check_event(self, event_a, event_b):
        self.current_event_ids[0] = event_a
        self.current_event_ids[1] = event_b

    ## 2. Data Cleansing & Pre-Processing
    def step(self, mol_a, mol_b):
        n_mols = np.zeros(2)

        # Before the 23rd day, number of mols is zero for both line a and line b
        n_mols[0] = mol_a if self.current_time_idx >= 23 * 24 else 0.0
        n_mols[1] = mol_b if self.current_time_idx >= 23 * 24 else 0.0

        for line_idx in range(len(self.current_event_ids)):
            if self.current_event_ids[line_idx] == EVENT_CHECK_1:
                if self.process[line_idx] == PROCESS_YES:
                    self.process[line_idx] = PROCESS_NO
                    self.check_time[line_idx] = CHECK_TIME

                self.check_time[line_idx] -= 1
                self.check_mode[line_idx] = CHECK_MODE_1

                if self.check_time[line_idx] == 0:
                    self.process[line_idx] = PROCESS_YES
                    self.process_time[line_idx] = 0

                n_mols[:] = 0.0
            elif self.current_event_ids[line_idx] == EVENT_CHECK_2:
                if self.process[line_idx] == PROCESS_YES:
                    self.process[line_idx] = PROCESS_NO
                    self.check_time[line_idx] = CHECK_TIME

                self.check_time[line_idx] -= 1
                self.check_mode[line_idx] = CHECK_MODE_2

                if self.check_time[line_idx] == 0:
                    self.process[line_idx] = PROCESS_YES
                    self.process_time[line_idx] = 0

                n_mols[:] = 0.0

            elif self.current_event_ids[line_idx] == EVENT_CHECK_3:
                if self.process[line_idx] == PROCESS_YES:
                    self.process[line_idx] = PROCESS_NO
                    self.check_time[line_idx] = CHECK_TIME

                self.check_time[line_idx] -= 1
                self.check_mode[line_idx] = CHECK_MODE_3

                if self.check_time[line_idx] == 0:
                    self.process[line_idx] = PROCESS_YES
                    self.process_time[line_idx] = 0

                n_mols[:] = 0.0

            elif self.current_event_ids[line_idx] == EVENT_CHECK_4:
                if self.process[line_idx] == PROCESS_YES:
                    self.process[line_idx] = PROCESS_NO
                    self.check_time[line_idx] = CHECK_TIME

                self.check_time[line_idx] -= 1
                self.check_mode[line_idx] = CHECK_MODE_4

                if self.check_time[line_idx] == 0:
                    self.process[line_idx] = PROCESS_YES
                    self.process_time[line_idx] = 0

                n_mols[:] = 0.0

            elif self.current_event_ids[line_idx] == EVENT_PROCESS:
                self.process_time[line_idx] += 1

                if self.process_time[line_idx] == MIN_PROCESS_TIME:
                    self.process[line_idx] = PROCESS_NO
                    self.check_time[line_idx] = CHECK_TIME

                # Update mol
                remove_list = []
                self.inqueue_mols[line_idx, :] = 0.0
                for i, mol in enumerate(self.running_mols[line_idx]):
                    mol_name = mol['name']
                    mol_value = mol['value']

                    mol['process_time'] += 1
                    # If the process time has reached 48, parts now beocome mol with a fixed molding rate 
                    if mol['process_time'] == 48:
                        self.running_stock.loc[self.current_time_idx, mol_name] += mol_value * MOLDING_RATE
                        remove_list.append(i)
                    # Parts that didn't become mols, go into inqueue_mols: waiting to be processed to be mols
                    else:
                        self.inqueue_mols[line_idx, int(mol_name[-1]) - 1] += mol_value

                self.running_mols[line_idx] = [item for i, item in enumerate(self.running_mols[line_idx])
                                               if i not in remove_list]

                if n_mols[line_idx] > 0:
                    # Estimate the part needed
                    self.running_stock.loc[self.current_time_idx, 'PRT_{}'.format(self.check_mode[line_idx] + 1)] += - \
                        n_mols[line_idx]

                    # Add mol
                    new_mol = dict(
                        name='MOL_{}'.format(self.check_mode[line_idx] + 1),
                        value=n_mols[line_idx],
                        process_time=0
                    )
                    self.running_mols[line_idx].append(new_mol)
                    self.inqueue_mols[line_idx, self.check_mode[line_idx]] += new_mol['value']
            else:
                raise Exception("Something was wrong!")

        if self.current_time_idx > 0:
            self.running_stock.loc[self.current_time_idx] += self.running_stock.loc[self.current_time_idx - 1]

        current_time = submission_df.loc[self.current_time_idx, 'time']
        current_order = order_df.loc[self.current_order_idx, 'time']

        # If the time of the order and the current time is the same, process the order and
        # calculate the differences in block created and block ordered
        if current_order == current_time:
            for column in ['BLK_1', 'BLK_2', 'BLK_3', 'BLK_4']:
                val = order_df.loc[self.current_order_idx, column]
                if val > 0:
                    self.N[int(column[-1]) - 1] += val

                    mol_col = BLK_TO_MOL[column]
                    mol_num = self.running_stock.loc[self.current_time_idx, mol_col]
                    self.running_stock.loc[self.current_time_idx, mol_col] = 0

                    # Generating blocks with appropriate cut rates
                    blk_gen = int(mol_num * get_cut_rate(blk=column, month=current_time.month) * MAX_CUT[column])
                    blk_stock = self.running_stock.loc[self.current_time_idx, column] + blk_gen
                    blk_diff = blk_stock - val

                    self.running_stock.loc[self.current_time_idx, column] = blk_diff

                    # Assigning values for q and p such that they can be later used to calculate reward/score.
                    if blk_diff > 0:
                        self.q[int(column[-1]) - 1] += blk_diff
                    else:
                        self.p[int(column[-1]) - 1] += abs(blk_diff)

            self.current_order_idx += 1

        part_1 = self.running_stock.loc[self.current_time_idx, 'PRT_1']
        part_2 = self.running_stock.loc[self.current_time_idx, 'PRT_2']
        part_3 = self.running_stock.loc[self.current_time_idx, 'PRT_3']
        part_4 = self.running_stock.loc[self.current_time_idx, 'PRT_4']
        envInfo = EnvInfo(current_time_idx=self.current_time_idx,
                          event_a=self.current_event_ids[0],
                          mol_a=n_mols[0],
                          event_b=self.current_event_ids[1],
                          mol_b=n_mols[1],
                          part_1=part_1,
                          part_2=part_2,
                          part_3=part_3,
                          part_4=part_4)

        if self.current_time_idx >= 2183:
            done = True
        else:
            done = False

        # Update next step
        for line_idx in range(2):
            if self.process[line_idx] == PROCESS_YES:
                self.current_event_ids[line_idx] = EVENT_PROCESS

        self.current_time_idx += 1

        # Obtain next observation
        next_check_obs, next_process_obs = self._get_obs()

        return next_check_obs, next_process_obs, done, envInfo

    # Reward function for the check event model, only dealing with blk1, blk3, blk4
    def get_check_reward(self):
        check_reward = 0.0

        # Reward function that give highest score
        for ii in [0, 2, 3]:
            check_reward += (10.0 * (np.sum(self.N[ii])) - (
                    0.5 * (np.sum(self.p[ii])) + 0.2 * (np.sum(self.q[ii])))) / (10.0 * (np.sum(self.N[ii])))

        # This is general reward function (but it doesn't reach the highest score)
        # check_reward = (10.0 * (np.sum(self.N)) - (
        #         0.5 * (np.sum(self.p)) + 0.2 * (np.sum(self.q)))) / (10.0 * (np.sum(self.N)))

        return check_reward

    # Get reward value for each process model (blk1~4)
    def get_process_reward(self):
        process_r = (10.0 * self.N -
                     (0.5 * self.p + 0.2 * self.q)) / \
                    (10.0 * self.N)
        return process_r

    # Get Dacon Score
    def get_dacon_score(self):
        return (10.0 * (np.sum(self.N)) - (
                0.5 * (np.sum(self.p)) + 0.2 * (np.sum(self.q)))) / (10.0 * (np.sum(self.N)))

    # Return state for each model
    def _get_obs(self):
        # state of check model
        event_order_info = order_df.loc[
                           self.current_time_idx // 24:(self.current_time_idx // 24 + self.check_offset_days),
                           'BLK_1':'BLK_4'].values

        if self.current_time_idx == 0:
            running_stock_info = self.running_stock.loc[self.current_time_idx].values
        else:
            running_stock_info = self.running_stock.loc[self.current_time_idx - 1].values

        check_obs = np.concatenate([event_order_info.reshape(-1),  # order info
                                    running_stock_info[4:8] * 400,  # MOL info
                                    running_stock_info[8:],  # BLK info
                                    np.sum(self.inqueue_mols, axis=0) * 400  # inqueue info
                                    ]).astype(np.float32)

        mol_order_info = order_df.loc[
                         self.current_time_idx // 24:(self.current_time_idx // 24 + self.process_offset_days),
                         'BLK_1':'BLK_4'].values

        # state of process models
        process_obs = []
        for ii in range(4):
            process_obs_ii = np.concatenate([mol_order_info[:, ii].reshape(-1),  # order info
                                             [running_stock_info[ii + 4] * 400],  # MOL info
                                             [running_stock_info[ii + 8]],  # BLK info
                                             [np.sum(self.inqueue_mols[:, ii]) * 400],  # inqueue info
                                             [(self.check_mode[0] == self.check_mode[1]) * 1000000]
                                             # process in both line or not: 1 for both lines, 0 for one line
                                             ]).astype(np.float32)
            process_obs.append(process_obs_ii / 1000000.0)  # Normalizing

        # Normalizing
        return check_obs / 1000000.0, process_obs


# Function to make mlp neural net
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        acti = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), acti()]
    return nn.Sequential(*layers)


# Actor network
class Actor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


# Value Network
class Critic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)


# Proximal Policy Optimization reinforcement learning algorithm
class PPO(nn.Module):

    def __init__(self,
                 observation_dim,
                 action_dim,
                 hidden_sizes=(64, 64),
                 activation=nn.Tanh,
                 buff_length=2184,
                 learning_rate=3e-4,
                 train_iters=80,
                 clipping_ratio=0.2,
                 discount=0.99,
                 lamb=0.95
                 ):
        super().__init__()

        # policy builder depends on action space
        self.pi = Actor(observation_dim, action_dim, hidden_sizes, activation)

        # build value function
        self.v = Critic(observation_dim, hidden_sizes, activation)

        # Contain data for training at each epoch
        self.buf = Buffer(observation_dim, action_dim, buff_length, discount, lamb)

        # Set up optimizers for policy and value function
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.train_iters = train_iters
        self.clip_ratio = clipping_ratio

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

    def update(self):
        # Obtain data from Buffer
        data = self.buf.get()

        obs, act, adv, logp_old, ret = data['obs'], data['act'], data['adv'], data['logp'], data['ret']

        # Train policy and value function with multiple steps of gradient descent
        for it in range(self.train_iters):
            self.optimizer.zero_grad()

            # Policy loss
            pi, logp = self.pi(obs, act)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            pi_loss = -(torch.min(ratio * adv, clip_adv)).mean()

            # Value loss
            value_loss = 0.5 * ((self.v(obs) - ret) ** 2).mean()

            # Loss
            loss = pi_loss + value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()


# Class for storing information
class Buffer:
    """
    A buffer for storing samples
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)  # observation (state)
        self.act_buf = np.zeros(size, dtype=np.float32)  # action
        self.adv_buf = np.zeros(size, dtype=np.float32)  # advantage
        self.rew_buf = np.zeros(size, dtype=np.float32)  # reward
        self.ret_buf = np.zeros(size, dtype=np.float32)  # target value
        self.val_buf = np.zeros(size, dtype=np.float32)  # value
        self.logp_buf = np.zeros(size, dtype=np.float32)  # log probability
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size  # internal index

    def store(self, obs, act, rew, val, logp):
        """
        Store a single transition (state, action, reward, value, log prob) to the buffer
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def store_from_other_buffer(self, buffer):
        """
        Store th whole episode from another buffer
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.path_start_idx:self.path_start_idx + buffer.ptr] = buffer.obs_buf[:buffer.ptr]
        self.act_buf[self.path_start_idx:self.path_start_idx + buffer.ptr] = buffer.act_buf[:buffer.ptr]
        self.rew_buf[self.path_start_idx:self.path_start_idx + buffer.ptr] = buffer.rew_buf[:buffer.ptr]
        self.val_buf[self.path_start_idx:self.path_start_idx + buffer.ptr] = buffer.val_buf[:buffer.ptr]
        self.logp_buf[self.path_start_idx:self.path_start_idx + buffer.ptr] = buffer.logp_buf[:buffer.ptr]

        self.adv_buf[self.path_start_idx:self.path_start_idx + buffer.ptr] = buffer.adv_buf[:buffer.ptr]
        self.ret_buf[self.path_start_idx:self.path_start_idx + buffer.ptr] = buffer.ret_buf[:buffer.ptr]

        # Update internal index
        self.path_start_idx += buffer.ptr
        self.ptr += buffer.ptr

    # This function calculate advantage and target value for each episode
    def finish_path(self, last_val=0, last_reward=None):
        if last_reward is not None:
            self.rew_buf[self.ptr - 1] = last_reward

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE-Lambda advantage calculation
        td_target = rews[:-1] + self.gamma * vals[1:]
        td_error = td_target - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(td_error, self.gamma * self.lam)
        self.ret_buf[path_slice] = self.adv_buf[path_slice] + vals[:-1]

        # update internal index
        self.path_start_idx = self.ptr

    def get(self):
        """
        get data from buffer and reset internal index
        """
        data = dict(obs=self.obs_buf[:self.ptr, :], act=self.act_buf[:self.ptr], ret=self.ret_buf[:self.ptr],
                    adv=self.adv_buf[:self.ptr], logp=self.logp_buf[:self.ptr])

        self.ptr, self.path_start_idx = 0, 0

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def reset(self):
        self.ptr, self.path_start_idx = 0, 0


# Collecting data using Joblib parallel
def data_collector(check_model, process_models, environment, toySubmitter, epoch, itr, train_mode):
    # Make a copy version of environemt and submitter
    # to make sure their changes does not affect to the ones from main thread
    local_env = deepcopy(environment)
    local_submitter = deepcopy(toySubmitter)

    # We make a copy of sample buffer depending on the train mode
    if train_mode != -1:
        if train_mode == 0:
            buffer = deepcopy(check_model.buf)
        else:
            buffer = deepcopy(process_models[train_mode - 1].buf)
        buffer.reset()
    else:
        buffer = None

    # Get initial check event from file
    def get_initial_check_from_file():
        check_a = int(local_submitter.submission.loc[0, "Event_A"][-1]) - 1
        check_b = int(local_submitter.submission.loc[0, "Event_B"][-1]) - 1

        return check_a, check_b

    # In case of inference or training process models, make a fix schedule first
    if train_mode != 0:
        check_o, _ = local_env.reset()
        for ep_len in range(2184):
            current_day = submission_df.loc[ep_len, 'time']
            a1 = check_model.act(torch.as_tensor(check_o, dtype=torch.float32))
            event_a = local_env.check_actions[a1.item()]['check_a']
            event_b = local_env.check_actions[a1.item()]['check_b']
            a2_a = a2_b = get_max_process(-1, current_day.month)

            modulo_126 = ep_len % 126
            if modulo_126 == 0:
                local_env.change_check_event(event_a, event_b)
            next_event_o, _, d, env_info = local_env.step(mol_a=a2_a, mol_b=a2_b)
            local_submitter.step(env_info)
            # Update step
            check_o = next_event_o

    # Prepare for interaction with environment
    # (reset environment, update the initial check event)
    check_o, process_o = local_env.reset()
    init_check_a, init_check_b = get_initial_check_from_file()
    local_env.set_initial_check_mode(init_check_a, init_check_b)

    # Start collecting datasets
    for ep_len in range(2184):
        current_day = submission_df.loc[ep_len, 'time']

        # Inference
        if train_mode == -1:
            # Get event from a fixed schedule that has been created above
            try:
                event_a = int(local_submitter.submission.loc[ep_len, "Event_A"][-1]) - 1
                event_b = int(local_submitter.submission.loc[ep_len, "Event_B"][-1]) - 1
            except Exception:
                # PROCESS
                event_a = None
                event_b = None
            if local_env.check_mode[0] == local_env.check_mode[1]:
                a2_pt = process_models[local_env.check_mode[0]].act(
                    torch.as_tensor(process_o[local_env.check_mode[0]],
                                    dtype=torch.float32))
                a2_a = a2_b = get_max_process(local_env.check_mode[0], current_day.month) \
                    if a2_pt.item() == 1 else 0.0
            else:
                a2_a_pt = process_models[local_env.check_mode[0]].act(
                    torch.as_tensor(process_o[local_env.check_mode[0]],
                                    dtype=torch.float32))
                a2_a = get_max_process(local_env.check_mode[0], current_day.month) \
                    if a2_a_pt.item() == 1 else 0.0

                a2_b_pt = process_models[local_env.check_mode[1]].act(
                    torch.as_tensor(process_o[local_env.check_mode[1]],
                                    dtype=torch.float32))
                a2_b = get_max_process(local_env.check_mode[1], current_day.month) \
                    if a2_b_pt.item() == 1 else 0.0
        # Collect data for check model
        elif train_mode == 0:
            # Get actions from check model
            a1, v1, logp1 = check_model.step(torch.as_tensor(check_o, dtype=torch.float32))
            event_a = local_env.check_actions[a1.item()]['check_a']
            event_b = local_env.check_actions[a1.item()]['check_b']

            # Process with maximum of mol if possible
            a2_a = get_max_process(-1, current_day.month)
            a2_b = get_max_process(-1, current_day.month)
        # Collect data for process model
        else:
            # Get event from a fixed schedule from the check model
            try:
                event_a = int(local_submitter.submission.loc[ep_len, "Event_A"][-1]) - 1
                event_b = int(local_submitter.submission.loc[ep_len, "Event_B"][-1]) - 1
            except Exception:
                # PROCESS
                event_a = None
                event_b = None

            # Get action from process model.
            # Only use the process model of the current train_mode
            # If both lines are "checking" the same block ex) Line A: check1 Line B: check1
            if local_env.check_mode[0] == local_env.check_mode[1]:
                # Depend on the current train_mode, we get action from model or from file (deterministic)
                deterministic = False
                if train_mode - 1 != local_env.check_mode[0]:
                    deterministic = True

                # Get action from current trained model or a fix value
                process_model = process_models[local_env.check_mode[0]]
                if deterministic:
                    a2_a = a2_b = float(local_submitter.submission.loc[ep_len, "MOL_A"])
                else:
                    a2_pt, v2, logp2 = process_model.step(torch.as_tensor(process_o[local_env.check_mode[0]],
                                                                          dtype=torch.float32))
                    a2_a = a2_b = get_max_process(local_env.check_mode[0], current_day.month) \
                        if a2_pt.item() == 1 else 0.0

            # In case line_a and line_b process different block ex) Line A: check2 Line B: check3
            else:
                # Similarly, depend on the current train_mode, we get action from model or
                # from file (deterministic)
                deterministic_a = False
                deterministic_b = False
                # If one line's check event is not the same as the block that is being trained
                # ex) check1, train_mode = 2
                if train_mode - 1 != local_env.check_mode[0]:
                    deterministic_a = True

                # If other line's check event is not the same as the block that is being trained
                if train_mode - 1 != local_env.check_mode[1]:
                    deterministic_b = True

                process_model_a = process_models[local_env.check_mode[0]]
                if deterministic_a:
                    a2_a = float(local_submitter.submission.loc[ep_len, "MOL_A"])
                else:
                    a2_a_pt, v2_a, logp2_a = process_model_a.step(torch.as_tensor(process_o[local_env.check_mode[0]],
                                                                                  dtype=torch.float32))
                    a2_a = get_max_process(local_env.check_mode[0], current_day.month) \
                        if a2_a_pt.item() == 1 else 0.0
                process_model_b = process_models[local_env.check_mode[1]]
                if deterministic_b:
                    a2_b = float(local_submitter.submission.loc[ep_len, "MOL_B"])
                else:
                    a2_b_pt, v2_b, logp2_b = process_model_b.step(torch.as_tensor(process_o[local_env.check_mode[1]],
                                                                                  dtype=torch.float32))
                    a2_b = get_max_process(local_env.check_mode[1], current_day.month) \
                        if a2_b_pt.item() == 1 else 0.0

        # Get observation and transit to the next state; each state is each group of time (28+98 = 126)
        modulo_126 = ep_len % 126
        if modulo_126 == 0:
            local_env.change_check_event(event_a, event_b)
        next_event_o, next_process_o, d, env_info = local_env.step(mol_a=a2_a, mol_b=a2_b)

        # Store transition to buffer
        # For check mode, storing one "group" of data
        if train_mode != -1:
            if train_mode == 0:
                # For check model, we only store transitions from the 5th check event (Ignore first four check events)
                if modulo_126 == 0 and ep_len >= 126 * 4:
                    buffer.store(check_o, a1, 0.0, v1, logp1)
            else:
                # For process model only stores transitions from 04/24.
                if modulo_126 >= 28 and ep_len >= 23 * 24:
                    if local_env.check_mode[0] == local_env.check_mode[1]:
                        if train_mode - 1 == local_env.check_mode[0]:
                            buffer.store(process_o[train_mode - 1], a2_pt, 0.0, v2, logp2)
                    else:
                        if train_mode - 1 == local_env.check_mode[0]:
                            buffer.store(process_o[train_mode - 1], a2_a_pt, 0.0, v2_a, logp2_a)
                        elif train_mode - 1 == local_env.check_mode[1]:
                            buffer.store(process_o[train_mode - 1], a2_b_pt, 0.0, v2_b, logp2_b)

        # Update csv file for submission
        local_submitter.step(env_info)

        # Update step
        process_o = next_process_o
        check_o = next_event_o

        # 6. Conclusion & Discussion
        # Episode done, get the episode reward and calculate advantage$value of episode
        if d:
            check_reward = local_env.get_check_reward()
            process_rewards = local_env.get_process_reward()
            dacon_score = local_env.get_dacon_score()
            if train_mode != -1:
                # If check mode, last reward is the check reward  of the environment
                if train_mode == 0:
                    buffer.finish_path(last_reward=check_reward)
                # If process mode, last reward is the process reward  of the environment
                else:
                    buffer.finish_path(last_reward=process_rewards[train_mode - 1])

            # Update the submission file
            local_submitter.terminate(dacon_score, check_reward, process_rewards, train_mode, epoch, itr)

    # Return buffer and submitter to main thread
    return buffer, local_submitter


## 5. Model Tuning & Evaluation
def run(train_mode=0):
    # get parameters from the command line
    params = vars(get_params())
    logger.info(params)

    log_path = params['log_path']
    seed = params['seed']
    clip_ratio = params['clip_ratio']
    discount_factor = params['discount_factor']
    gae_lambda = params['gae_lambda']
    learning_rate = params['learning_rate']
    train_iters = params['train_iters']

    hidden_sizes = params['hidden_sizes']

    # Deciding activation type
    if params['activation'] == 'tanh':
        activation = torch.nn.Tanh
    elif params['activation'] == 'relu':
        activation = torch.nn.ReLU
    elif params['activation'] == 'leaky_relu':
        activation = torch.nn.LeakyReLU
    else:
        activation = torch.nn.Tanh

    event_offset_days = params['check_offset_days']
    process_offset_days = params['process_offset_days']

    # Parameters for running the code
    save_interval = params['save_interval']
    n_episode_per_epoch = params['n_episode_per_epoch']
    n_spent_epoch = params['n_spent_epoch'][train_mode] if train_mode != -1 else 1

    # Instantiate environment
    env = ToyEnv(check_offset_days=event_offset_days, process_offset_days=process_offset_days)
    toySubmitter = ToySubmitter(log_path=log_path)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create actor-critic model
    # [28, 98] * 17 + [42]
    # For check model, we are dealing with 14 "groups" of "check/process"
    # First four "groups" don't matter
    event_buff_length = (18 - 4) * n_episode_per_epoch

    # PPO for check_model
    check_model = PPO(observation_dim=env.check_obs_dim,
                      action_dim=len(env.check_actions),
                      hidden_sizes=hidden_sizes,
                      buff_length=event_buff_length,
                      learning_rate=learning_rate,
                      train_iters=train_iters,
                      clipping_ratio=clip_ratio,
                      activation=activation,
                      discount=discount_factor,
                      lamb=gae_lambda)

    process_buff_length = (78 + 98 * 12 + 14) * n_episode_per_epoch
    process_models = []

    # PPO for block model
    for ii in range(4):  # PROCESS BLK 1,2,3,4
        process_model = PPO(observation_dim=env.process_obs_dim,
                            action_dim=len(env.process_actions),
                            hidden_sizes=hidden_sizes,
                            buff_length=process_buff_length,
                            learning_rate=learning_rate,
                            train_iters=train_iters,
                            clipping_ratio=clip_ratio,
                            activation=activation,
                            discount=discount_factor,
                            lamb=gae_lambda)
        process_models.append(process_model)

    # Calling check points, if applicable
    check_model_checkpoint = params['check_model_checkpoint']
    blk_1_process_model_checkpoint = params['blk_1_process_model_checkpoint']
    blk_2_process_model_checkpoint = params['blk_2_process_model_checkpoint']
    blk_3_process_model_checkpoint = params['blk_3_process_model_checkpoint']
    blk_4_process_model_checkpoint = params['blk_4_process_model_checkpoint']
    pre_trained_models = {
        "check_model": f"check_model_{check_model_checkpoint}.pt",
        "blk_1_process_model": f"blk_1_process_model_{blk_1_process_model_checkpoint}.pt",
        "blk_2_process_model": f"blk_2_process_model_{blk_2_process_model_checkpoint}.pt",
        "blk_3_process_model": f"blk_3_process_model_{blk_3_process_model_checkpoint}.pt",
        "blk_4_process_model": f"blk_4_process_model_{blk_4_process_model_checkpoint}.pt"
    }
    for key, model_file in pre_trained_models.items():
        if os.path.exists(f"{log_path}/{model_file}"):
            print(f"model {key} loaded!")
            checkpoint = torch.load(f"{log_path}/{model_file}")
            if key == "check_model":
                check_model.load_state_dict(checkpoint[f"{key}"])

                if train_mode == -1:
                    check_model.eval()
            else:
                process_models[int(key[4]) - 1].load_state_dict(checkpoint[f"{key}"])
                if train_mode == -1:
                    process_models[int(key[4]) - 1].eval()

    # Main loop: collect experience in env and update/log each epoch
    for epoch in itertools.count():
        print(f"Collecting {n_episode_per_epoch} episodes at epoch {epoch}...")

        # Collecting many data sets using parallel
        with Parallel(n_jobs=int(cpu_count())) as parallel:
            results = parallel(
                delayed(data_collector)(check_model, process_models, env, toySubmitter, epoch, it, train_mode)
                for it in range(n_episode_per_epoch))

        # Append local buffer to main buffer
        sum_evaluate_reward = 0
        for result in results:
            localBuffer = result[0]
            localSubmitter = result[1]
            if train_mode != -1:
                if train_mode == 0:
                    check_model.buf.store_from_other_buffer(localBuffer)
                else:
                    process_models[train_mode - 1].buf.store_from_other_buffer(localBuffer)

            # Find the best explored submission
            sum_evaluate_reward += localSubmitter.evaluate_reward
            if toySubmitter.best_reward[train_mode] < localSubmitter.best_reward[train_mode]:
                toySubmitter.best_reward[:] = localSubmitter.best_reward[:]
                toySubmitter.submission = localSubmitter.submission.copy()
            if toySubmitter.best_dacon_score < localSubmitter.best_dacon_score:
                toySubmitter.best_dacon_score = localSubmitter.best_dacon_score

        # Print average reward (average reward of current trained model)
        current_reward = ''
        if train_mode == -1:
            current_reward = "Dacon Score"
        elif train_mode == 0:
            current_reward = "Check Reward"
        else:
            current_reward = f"BLK_{train_mode} Reward"
        print_log = f"Epoch: {epoch} | " \
                    f"Average {current_reward}: {sum_evaluate_reward / n_episode_per_epoch}"
        toySubmitter.log(print_log)

        # Return if inference mode
        if train_mode == -1:
            break

        # Perform PPO update!
        print(f"Optimizing model at epoch {epoch}")
        if train_mode == 0:
            check_model.update()
        else:
            if process_models[train_mode - 1].buf.ptr > 0:
                process_models[train_mode - 1].update()

        if epoch % save_interval == 0:
            if train_mode == 0:
                torch.save({"check_model": check_model.state_dict()},
                           f"{log_path}/check_model_{epoch}.pt")
            else:
                torch.save({f"blk_{train_mode}_process_model": process_models[train_mode - 1].state_dict()},
                           f"{log_path}/blk_{train_mode}_process_model_{epoch}.pt")

        # train other models
        n_spent_epoch -= 1
        if n_spent_epoch == 0:
            train_mode += 1
            if train_mode > 4:
                break
            n_spent_epoch = params['n_spent_epoch'][train_mode]
            print(f"Start to train BLK{train_mode} process model")


if __name__ == '__main__':
    # Parameters for running PPO
    def get_params():
        ''' Get parameters from command line '''
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=123)
        parser.add_argument("--log_path", type=str,
                            default=os.path.join(Path(__file__).resolve().parent, f"submissions"))
        parser.add_argument("--clip_ratio", type=float, default=0.1)
        parser.add_argument("--discount_factor", type=float, default=0.99)
        parser.add_argument("--gae_lambda", type=float, default=0.95)
        parser.add_argument("--learning_rate", type=float, default=0.00025)
        parser.add_argument("--train_iters", type=int, default=100)
        parser.add_argument("--hidden_sizes", type=list, default=[256, 256])
        parser.add_argument("--activation", type=str, default='tanh')
        parser.add_argument("--save_interval", type=int, default=1)
        parser.add_argument("--n_episode_per_epoch", type=int, default=100)
        parser.add_argument("--n_spent_epoch",
                            help="number of epoch to train each model",
                            type=list,
                            default=[60, 40, 40, 40, 40])
        parser.add_argument("--check_offset_days", type=int, default=30)
        parser.add_argument("--process_offset_days", type=int, default=30)
        parser.add_argument("--check_model_checkpoint", type=int, default=None)
        parser.add_argument("--blk_1_process_model_checkpoint", type=int, default=None)
        parser.add_argument("--blk_2_process_model_checkpoint", type=int, default=None)
        parser.add_argument("--blk_3_process_model_checkpoint", type=int, default=None)
        parser.add_argument("--blk_4_process_model_checkpoint", type=int, default=None)

        args, _ = parser.parse_known_args()
        return args


    # Parameter for deciding which model to train
    # -1: inference 0: check model, 1: blk_1 model, 2: blk_2 model, 3: blk_3 model, 4: blk_4 model
    run(train_mode=0)