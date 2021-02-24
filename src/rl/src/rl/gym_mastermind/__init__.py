#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym.envs.registration import register


register(
    id='mastermind-v0',
    entry_point='rl.gym_mastermind.envs:MastermindEnv'
)