#!/usr/bin/env python
# -*- coding: utf-8 -*-


# exceptions-----------------------------------------------------------------

class InvalidAlphaError(ValueError):

    def __init__(self, alpha, message='alpha must be in [0, 1]'):
        self.alpha = alpha
        self.message = message
        super(ValueError, self).__init__(self.message)

    def __str__(self):
        return '\'{alpha}\' -> {message}'.format(alpha=self.alpha, message=self.message)



class InvalidGammaError(ValueError):

    def __init__(self, gamma, message='gamma must be in [0, 1]'):
        self.gamma = gamma
        self.message = message
        super(ValueError, self).__init__(self.message)

    def __str__(self):
        return '\'{gamma}\' -> {message}'.format(gamma=self.gamma, message=self.message)



class InvalidEpsilonError(ValueError):

    def __init__(self, epsilon, message='epsilon must be in [0, 1]'):
        self.epsilon = epsilon
        self.message = message
        super(ValueError, self).__init__(self.message)

    def __str__(self):
        return '\'{epsilon}\' -> {message}'.format(epsilon=self.epsilon, message=self.message)



class InvalidEpsilonModeError(ValueError):

    def __init__(self, epsilon_mode, epsilon_modes, message='epsilon_mode must be one of '):
        self.epsilon_mode = epsilon_mode
        self.message = message + str(epsilon_modes)
        super(ValueError, self).__init__(self.message)

    def __str__(self):
        return '\'{epsilon_mode}\' -> {message}'.format(epsilon_mode=self.epsilon_mode, message=self.message)



class InvalidActionError(ValueError):

	def __init__(self, action, message='invalid action'):
		self.action = action
		self.message = message
		super(ValueError, self).__init__(self.message)

	def __str__(self):
		return '\'{action}\' -> {message}'.format(action=self.action, message=self.message)



class InvalidStateError(ValueError):

	def __init__(self, state, message='invalid state'):
		self.state = state
		self.message = message
		super(ValueError, self).__init__(self.message)

	def __str__(self):
		return '\'{state}\' -> {message}'.format(state=self.state, message=self.message)



class InvalidSecretError(Exception):

    def __init__(self, message='secret length cannot be 0'):
        self.message = message
        super(Exception, self).__init__(self.message)



class UnsupportedOperationError(Exception):

    def __init__(self, message='unsupported operation'):
        self.message = message
        super(Exception, self).__init__(self.message)
