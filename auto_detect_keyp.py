import glob
import os
import pickle
from itertools import islice

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte
from skimage.transform import rotate

import hyperparameters
import train_keyp_inverse_forward
import train_keyp_pred
import train_dynamics
import utils

import matplotlib.pyplot as plt
import gym

from visualizer import viz_track, viz_imgseq_goal, viz_imgseq

import robosuite as suite


class MPC:
    def __init__(self, model, goal_state, action_dim=4, H=50):
        """

        :param model: f(s_t, a_t) -> del(s_t) ; s_{t+1} = s_t + del(s_t)
        :param state_dim: = 2*num_keyp
        :param action_dim: action_dim
        """
        self.model = model

        self.H = H
        self.num_sample_seq = 1000
        self.goal_state = goal_state
        self.action_dim = action_dim

    def predict_next_states(self, state, action):
        """

        :param state: N x num_keyp x 2
        :param action: N x T x action_dim
        :return: next_state: N x (T-1) x num_keyp x 2
        """
        next_states = self.model.keyp_pred_net.unroll(state, action)

        state = state[:, None, :, :]
        next_states = torch.cat((state, next_states), dim=1)

        return next_states

    def random_action:
    l, h = -1, 1
    actions_batch = (l-h) * torch.rand(self.num_sample_seq, self.H, self.action_dim) + h

    state_batch = state.unsqueeze(0)
    state_batch = state_batch.repeat((self.num_sample_seq, 1, 1))# N x N_K x 2

    def score(self, state_batch, goal_state):
        """

        :param state_batch: N x T x num_keyp x 2
        :param goal_state: num_keyp x 2
        :return: cost: (N,)
        """
        for i in range(T):
            # goal_state = goal_state[None, None, :, :]
            # curr_state = state_batch[:, :, :, :]
            goal_state = goal_state[[i], :]
            curr_state = state_batch[[i+1], :]
            T = state_batch.shape[1]
            cost = torch.sum((curr_state - goal_state)**2, dim=(1,2,3))/T

            #cost = torch.sum((curr_state - goal_state) ** 2, dim=(1, 2)) / T
        return cost

    def get_keyp_state(self, im):
        im = im[np.newaxis, :, :, :]
        im = convert_img_torch(im)
        keyp = self.model.img_to_keyp(im.unsqueeze(0))[0, 0, :, :2]
        return keyp

def auto_detect_key()