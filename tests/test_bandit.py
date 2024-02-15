import unittest
from multi_arm_bandit.bandit import TwoArmedBandit
from multi_arm_bandit.distributions import uniform_dist

class TestTwoArmedBandit(unittest.TestCase):
    def test_pull_arm(self):
        bandit = TwoArmedBandit(uniform_dist(0, 1), uniform_dist(0, 1), uniform_dist(0, 1), uniform_dist(0, 1))
        reward, punishment, weighted_sum = bandit.pull(0)
        self.assertTrue(0 <= reward <= 1)
        self.assertTrue(0 <= punishment <= 1)

if __name__ == '__main__':
    unittest.main()
