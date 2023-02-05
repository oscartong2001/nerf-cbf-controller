import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
import numpy as np
import math

import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

import argparse
import os
import time

import numpy as np
import torch
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader

from src import config
from src.tools.viz import SLAMFrontend
from src.utils.datasets import get_dataset
from src.utils.Renderer import Renderer
from src.NICE_SLAM import NICE_SLAM
from src.common import get_camera_from_tensor
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

with open('device.txt', encoding='utf-8') as file:
     device=file.read()
time_step = 0.1
u = torch.tensor([[0,0,1]]).t().float().to(device)
intend = 1


def state_to_pose(state):
    rm = torch.from_numpy(R.from_euler('xyz', state[3:].cpu(), degrees=True).as_matrix()).to(device)
    pose = torch.cat((torch.cat((rm, state[:3].unsqueeze(-1)), 1), torch.tensor([[0,0,0,1]]).to(device)), 0)
    return pose


def update_dynamics(state, action):
    return torch.cat((state[:6] + state[6:] * time_step + 0.5 * action  * (time_step ** 2), state[6:] + time_step * action))


class Robot:
    def __init__(self, cfg):
        self.c = np.load('replica_room1.npy',allow_pickle=True).item()
        self.decoders = torch.load('replica_room1.pth')
        self.renderer = NICE_SLAM(cfg, args).renderer
        self.renderer.H = 68
        self.renderer.W = 120
        self.renderer.fx = 60
        self.renderer.fy = 60
        self.renderer.cx = 59.95
        self.renderer.cy = 33.95
        self.v = torch.zeros(6).to(device)

    def predict_observation(self, pose):
        depth, _, _ = self.renderer.render_batch_img(
                    self.c,
                    self.decoders,
                    pose,
                    device,
                    stage='middle',
                    gt_depth=None)
        return depth
    
    def render(self, pose):
        self.renderer.H = 680
        self.renderer.W = 1200
        self.renderer.fx = 600.0
        self.renderer.fy = 600.0
        self.renderer.cx = 599.5
        self.renderer.cy = 339.5
        depth, _, _ = self.renderer.render_img(
                    self.c,
                    self.decoders,
                    pose,
                    device,
                    stage='color',
                    gt_depth=None)
        depth, uncertainty, color = self.renderer.render_img(
                    self.c,
                    self.decoders,
                    pose,
                    device,
                    stage='color',
                    gt_depth=depth)
        self.renderer.H = 68
        self.renderer.W = 120
        self.renderer.fx = 60
        self.renderer.fy = 60
        self.renderer.cx = 59.95
        self.renderer.cy = 33.95
        return depth, color
    
    def get_batch_density(self, points):
        return torch.tensor(abs(self.renderer.eval_points(
            points,
            self.decoders,
            self.c)[:, -1])).unsqueeze(-1).to(device)


class path(nn.Module):
    def __init__(self, init_actions, steps):
        super(path, self).__init__()
        self.steps = steps
        self.actions = nn.Parameter(torch.tensor(init_actions)).to(device)

    def forward(self, x):
        states = torch.zeros((self.steps, 12)).to(device)
        for iter in range(self.steps):
            if iter == 0:
                states[0] = update_dynamics(x, self.actions[0])
            else:
                states[iter] = update_dynamics(states[iter - 1], self.actions[iter])
        actions = self.actions
        return states, actions


class Planner:
    def __init__(self, start_pose, end_pose, conf, robot):
        self.robot = robot
        self.steps = conf["steps"]
        self.lr = conf["lr"]
        self.epochs_init = conf["epochs_init"]
        self.epochs_update = conf["epochs_update"]
        self.iter_number = conf["iter_number"]
        self.n_steps_to_execute = conf["n_steps_to_execute"]

        self.start_pose = start_pose
        self.end_pose = end_pose
        self.trajectory = self.start_pose.clone().reshape(1, -1).to(device)

        self.end_state_penalty = conf["end_state_penalty"]
        self.translation_penalty = conf["translation_penalty"]
        self.rotation_penalty = conf["rotation_penalty"]
        self.density_penalty = conf["density_penalty"]

    def plan_traj(self, state_estimate, init_actions, update=0):

        # update should be int describing the number of steps/actions already taken. update=0 means
        # we are initializing
        num_steps = self.steps - update

        if num_steps <= 0:
            print(
                "Planner cannot plan more steps. Please reinitialize planner with new waypoints."
            )
            return

        # Initialize path module
        starting_pose = torch.Tensor(state_estimate.float().cpu()).to(device)
        path_propagation = path(init_actions, num_steps)
        optimizer = torch.optim.Adam(
            params=path_propagation.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )

        if update == 0:
            num_iter = self.epochs_init
        else:
            num_iter = self.epochs_update

        for it in range(num_iter):
            optimizer.zero_grad()
            # t1 = time.time()
            projected_states, actions = path_propagation(starting_pose)
            # t2 = time.time()
            # print('Propagation', t2 - t1)

            loss = self.get_loss(projected_states, actions, num_steps)
            # t3 = time.time()
            # print('Calculating Loss', t3-t2)

            loss.backward()
            optimizer.step()

            # if it % 20 == 0:
                # print("Iteration", it)
                # print(projected_states[-1])
                # print("Loss", loss, loss.device)

            new_lrate = self.lr * (0.8 ** ((it + 1) / self.epochs_init))
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lrate

        #self.plot_trajectory(projected_states)

        return projected_states.detach(), actions.detach()

    def get_loss(self, states, actions, num_steps):
        density_loss = self.get_density_loss(states)
        state_loss = self.get_state_loss(states, num_steps)
        action_loss = self.get_action_loss(actions)

        loss = density_loss + state_loss + action_loss

        #print('Losses', density_loss, state_loss, action_loss)

        return loss

    def get_density_loss(self, states):

        next_states = torch.vstack((states, self.end_pose))
        prev_states = torch.vstack((self.start_pose, states))

        densities = self.robot.get_batch_density(states[:, :3]).to(device)

        distances = torch.norm(next_states[1:, :3] - prev_states[:-1, :3], dim=1, p=2).to(device)

        colision_prob = densities * distances[..., None].expand(*densities.shape)
        colision_prob = torch.mean(colision_prob, dim=1)

        density_loss = self.density_penalty * torch.sum(colision_prob)

        return density_loss

    def get_state_loss(self, states, num_steps):
        offsets = states - self.end_pose.expand(num_steps, -1)


        state_loss = torch.mean(
            (torch.norm(states[1:, ...] - states[:-1, ...], dim=1, p=2)) ** 2
        ) + 10 * torch.mean(torch.norm(offsets, dim=1, p=2) ** 2)

        state_loss = (
            state_loss
            + self.end_state_penalty
            * (torch.norm(states[-1, ...] - self.end_pose, p=2)) ** 2
        )

        return state_loss

    def get_action_loss(self, actions):
        action_loss = torch.mean(
            self.translation_penalty * (torch.norm(actions[:, :3], dim=1, p=2)) ** 2
            + self.rotation_penalty * (torch.norm(actions[:, 3:], dim=1, p=2)) ** 2
        )

        return action_loss

    def plan(self, start_pose, end_pose, intended_action):
        self.start_pose = start_pose
        self.end_pose = end_pose
        self.trajectory = self.start_pose.clone().reshape(1, -1).to(device)
        init_actions = torch.Tensor(self.steps, 6).uniform_(-1, 1).to(device)

        # Make an initial plan
        proj_states, action_planner = planner.plan_traj(self.start_pose, init_actions)

        # MPC LOOP
        current_state = self.start_pose
        for iter in range(self.iter_number):
            # Execute the first few steps of the planned trajectory, then replan
            for i in range(self.n_steps_to_execute):
                act_now = action_planner[i, :]
                current_state = update_dynamics(current_state, act_now)
                planner.trajectory = torch.vstack((planner.trajectory, current_state.reshape(1, -1)))
                if iter == 0 and i == 0:
                    action_list = act_now.unsqueeze(0)
                else:
                    action_list = torch.cat((action_list, act_now.unsqueeze(0)), dim=0)
            proj_states, action_planner = planner.plan_traj(current_state, action_planner, update=True)
            print('iter{}: intended action:{}\nreal action:{}'.format(iter, intended_action, torch.sum(action_list, dim=0).to(device)))
        return action_list


conf = {
    "steps": 10,
    "lr": 2e-3,
    "epochs_init": 100,
    "epochs_update": 100,
    "iter_number": 10,
    "end_state_penalty": 1e3,
    "translation_penalty": 1,
    "rotation_penalty": 1,
    "density_penalty": 1e3,
    "n_steps_to_execute": 5,
}
print(conf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Arguments to visualize the SLAM process.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one inconfig file')
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    parser.set_defaults(nice=True)
    parser.add_argument('--save_rendering',
                        action='store_true', help='save rendering video to `vis.mp4` in output folder ')
    parser.add_argument('--vis_input_frame',
                        action='store_true', help='visualize input frames')
    parser.add_argument('--no_gt_traj',
                        action='store_true', help='not visualize gt trajectory')
    args = parser.parse_args()
    
    cfg = config.load_config(
        args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')  # args.config: env, nice_slam.yaml: robot

    robot = Robot(cfg)
    planner = Planner(torch.zeros(12).to(device), torch.zeros(12).to(device), conf, robot)
    fps = 10
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #videoWriter = cv2.VideoWriter('video.avi',fourcc,fps,(240,68))
    #idx, color, depth, pose = frame_reader[0]
    pose = state_to_pose(torch.tensor([-2, 0.2,  0.5, 90, 0, 115]).to(device)).squeeze()
    depth, color = planner.robot.render(pose.to(device))
    #print('density:', robot.get_batch_density((pose[:-1,-1].unsqueeze(0).repeat(3,1)))[0])
    #videoWriter.write(np.hstack([cv2.normalize(depth.unsqueeze(-1).repeat(1,1,3).to(device).detach().cpu().numpy(),
    #dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8), (color[:, :, [2,1,0]]*255).astype(np.uint8).clip(0,255)]))
    #cv2.namedWindow("Safety Filter", cv2.WINDOW_KEEPRATIO)
    #cv2.imshow('Safety Filter', np.hstack([cv2.normalize(depth.unsqueeze(-1).repeat(1,1,3).to(device).detach().cpu().numpy(),
    #dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX), color[:, :, [2,1,0]].to(device).detach().cpu().numpy()]))

    #while True:
    for i in range(100):
        print('step:', i)
        #k = cv2.waitKeyEx()
        k=65362
        if k in {65362, 65364, 65361, 65363}:  # up, down, left, right
            
            if k == 65362:
                orient_action = torch.zeros(6).to(device)
                unit = -intend * torch.mm(pose[:3, :3].float(), u).squeeze().to(device)
                orient_action[0] = unit[0]
                orient_action[1] = unit[1]
                orient_action[2] = unit[2]
            elif k == 65364:
                orient_action = torch.zeros(6).to(device)
                unit = intend * torch.mm(pose[:3, :3].float(), u).squeeze().to(device)
                orient_action[0] = unit[0]
                orient_action[1] = unit[1]
                orient_action[2] = unit[2]
            elif k == 65361:
                orient_action[5] = intend * 10
            else:
                orient_action[5] = -intend * 10
            state = torch.cat((pose[:3, -1].to(device), torch.from_numpy(R.from_matrix(pose[:3, :3].cpu()).as_euler('xyz', degrees=True)).to(device)), dim=0).to(device)
            state = torch.cat((state, robot.v)).to(device)
            new_state = update_dynamics(state, orient_action)
            start = time.time()
            action_list = planner.plan(state, new_state, orient_action)
            end = time.time()
            #print(action_list)
            print('intended action: {}\nreal action:{}'.format(orient_action, torch.sum(action_list, dim=0).to(device)))
            print('intervention = {}, {}'.format(float(torch.norm(orient_action[:3] - torch.sum(action_list, dim=0)[:3].to(device), p=2)), float(torch.norm(orient_action[3:] - torch.sum(action_list, dim=0)[3:].to(device), p=2))))
            for i in range(action_list.shape[0]):
                new_state = update_dynamics(state, action_list[i])
                state = new_state
                robot.v = state[6:]
                pose = state_to_pose(state[:6])
            print('new_v = {}'.format(float(torch.norm(robot.v,p=2))))
            print('new density =', float(planner.robot.get_batch_density((pose[:-1,-1].unsqueeze(0).repeat(3,1)))[0]))
            pose = state_to_pose(state[:6])
            depth, color = planner.robot.render(pose.to(device))
            #cv2.imshow('Safety Filter', np.hstack([cv2.normalize(depth.unsqueeze(-1).repeat(1,1,3).to(device).detach().cpu().numpy(),
            #dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX), color[:, :, [2,1,0]].to(device).detach().cpu().numpy()]))
            print('new state = {}\nmin_depth = {}\ntime cost = {}'.format(state, depth.min(), end-start))
        elif k == 27:  # esc
            cv2.destroyAllWindows()
            break
