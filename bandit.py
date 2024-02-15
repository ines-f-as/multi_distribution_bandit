from .distributions import normal_dist, uniform_dist, gamma_dist, half_levy_dist, log_normal_dist, t_dist
import random
import numpy as np

class TwoArmedBandit:
    def __init__(self, arm1_reward_dist, arm1_punishment_dist, arm2_reward_dist, arm2_punishment_dist, reward_weight=1, punishment_weight=1):
        #initialize all the arm weights
        self.arm1_reward_dist = arm1_reward_dist
        self.arm1_punishment_dist = arm1_punishment_dist
        self.arm2_reward_dist = arm2_reward_dist
        self.arm2_punishment_dist = arm2_punishment_dist
        self.reward_weight = reward_weight
        self.punishment_weight = punishment_weight

    def pull(self, arm):
        if arm == 0:
            reward = self.arm1_reward_dist()
            punishment = self.arm1_punishment_dist()
        else:
            reward = self.arm2_reward_dist()
            punishment = self.arm2_punishment_dist()
        weighted_sum = self.reward_weight * reward - self.punishment_weight * punishment
        return reward, punishment, weighted_sum
