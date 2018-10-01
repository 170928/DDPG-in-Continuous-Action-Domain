import tensorflow as tf
import numpy as np
import network_model as ddpg
import gym
import os
from OU import Noise

def main():
    env = gym.make('Pendulum-v0')
    DDPG = ddpg(env)

if __name__ == "__main__":
    main()