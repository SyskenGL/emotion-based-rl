#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rl_exceptions as rlexc
import numpy as np
import itertools
import random
from collections import OrderedDict
from collections_extended import frozenbag


# constants------------------------------------------------------------------

EPSILON_MODES = [
	'e_greedy',
	'e_decaying'
]


# classes--------------------------------------------------------------------


class Agent:

	"""
	Rappresenta un agente con il compito di apprendere la sequenza vincente
	del gioco Mastermind (mediante IRL)

	Attributes
	-----------------------------------
	(MastermindEnv) env 
		Ambiente con cui l'agente interagisce

	(float) alpha 
			Indica il tasso di apprendimento

	(float) gamma
		Indica il tasso di sconto

	(float) epsilon
		Indica il tasso di exploitation/exploration

	(int) epsilon_mode
		Indica quale strategia adottare nella selezione delle azioni da eseguire.
		Sono supportate le strategie e_greedy e e_decay

	(int) epsilon_mode
		Indica quale strategia adottare nella selezione delle azioni da eseguire.
		Sono supportate le strategie e_greedy e e_decay

	(int) epsilon_mode
		Indica quale strategia adottare nella selezione delle azioni da eseguire.
		Sono supportate le strategie e_greedy e e_decay

	(float) epsilon_decay
		Indica il fattore di decadimento di epsilon

	(float) epsilon_low
		Indica il valore minimo consentito di epsilon

	(frozenbag) curr_state
		Indica lo stato attuale dell'agente in env

	(OrderedDict) qmatrix
		Indica la matrice Q dell'agente

	Methods
	-----------------------------------
	print_qmatrix()
		Stampa la matrice Q

	init_qmatrix()
		Inizializza la matrice Q

	update_qmatrix(reward)
		Aggiorna la matrice Q effettuando una propagazione della ricompensa all'indietro.
		La propagazione effettua l'aggiornamento considerando le azioni ottimali che possono
		essere intraprese in un certo stato s il quale può direttamente o indirettamente
		raggiungere lo stato finale a cui è stata assegnata la ricompensa

	take_action(action)
		Effettua l'azione passata in ingresso

	get_action_from_qmatrix()
		Restituisce l'azione da intraprendere. La scelta puo' essere di tipo exploration
		oppure exploitation, ciò dipende dal valore di epsilon al momento dell'invocazione
		del metodo.

	get_max_qvalue(state)
		Preleva il massimo valore Q dello stato passato in ingresso

	get_best_next_reachable_states(state)
		Restituisce la lista degli stati ottimali immediatamente successivi a quello
		passato in ingresso

	get_optimal()
		Restituisce la politica ottimale	
	"""

	def __init__(self, env, alpha=0.7, gamma=0.9, epsilon=0.999, epsilon_mode=EPSILON_MODES[1], epsilon_decay=0.95, epsilon_low=0.1):

		"""
		Parameters
		-----------------------------------
		(MastermindEnv) env 
			Ambiente con cui l'agente interagisce

		(float) alpha [opt, default = 0.7]
			Indica il tasso di apprendimento

		(float) gamma [opt, default = 0.9]
			Indica il tasso di sconto

		(float) epsilon [opt, default = 0.999]
			Indica il tasso di exploitation/exploration

		(int) epsilon_mode [opt, default = EPSILON_MODES[1]]
			Indica quale strategia adottare nella selezione delle azioni da eseguire.
			Sono supportate le strategie e_greedy e e_decay

		(float) epsilon_decay [opt, default = 0.95]
			Indica il fattore di decadimento di epsilon

		(float) epsilon_low [opt, default = 0.1]
			Indica il valore minimo consentito di epsilon

		Raises
		-----------------------------------
		InvalidAlphaError
			Il valore di alpha non è compreso in [0, 1]

		InvalidGammaError
			Il valore di gamma non è compreso in [0, 1]

		InvalidEpsilonError
			Il valore di epsilon e/o epsilon_low non è compreso in [0, 1]

		InvalidEpsilonModeError
			La strategia indicata non è supportata (non è indicata in EPSILON_MODES)
		"""

		if not 0 <= alpha <= 1:
		    raise rlexc.InvalidAlphaError(alpha)
		if not 0 <= gamma <= 1:
		    raise rlexc.InvalidGammaError(gamma)
		if not 0 <= epsilon <= 1:
		    raise rlexc.InvalidEpsilonError(epsilon)
		if not epsilon_mode in EPSILON_MODES:
		    raise rlexc.InvalidEpsilonModeError(epsilon_mode, EPSILON_MODES)
		if not 0 <= epsilon_low <= 1:
		    raise rlexc.InvalidEpsilonError(epsilon)

		self.env = env
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_mode = epsilon_mode
		self.epsilon_decay = epsilon_decay
		self.epsilon_low = epsilon_low
		self.curr_state = self.env.reset()
		self.qmatrix = self.init_qmatrix()


	def print_qmatrix(self):

		"""
		Stampa la matrice Q
		"""

		for state in self.qmatrix.keys():
			print '   {0:>{1}}: '.format('{' + str(list(state))[1:-1] + '}', 10) + str(self.qmatrix[state])


	def init_qmatrix(self):
		
		"""
		Inizializza la matrice Q
		"""

		states = self.env.get_states()
		qmatrix = OrderedDict()
		for state in states:
		    qmatrix[state] = np.full(self.env.action_space.n, 0, dtype = float)
		return qmatrix


	def update_qmatrix(self, reward):

		"""
		Aggiorna la matrice Q effettuando una propagazione della ricompensa all'indietro.
		La propagazione effettua l'aggiornamento considerando le azioni ottimali che possono
		essere intraprese in un certo stato s il quale può direttamente o indirettamente
		raggiungere lo stato finale a cui è stata assegnata la ricompensa

		Parameters
		-----------------------------------
		(float) reward
			Indica la ricompensa che l'utente ha assegnato allo stato (finale) raggiunto
		"""

		if self.env.is_terminal_state(self.curr_state):
			self.qmatrix[self.curr_state][:] = (
				self.qmatrix[self.curr_state][:]+self.alpha*(reward-self.qmatrix[self.curr_state][:])
			)
			coverage = self.env.get_coverage(self.curr_state)
			coverage.sort(key=len, reverse=True)
			for covered_state in coverage:
				best_reachable_states = self.get_best_next_reachable_states(covered_state)
				if len(best_reachable_states) != 0:
					for best_reachable_state in best_reachable_states:
						action = best_reachable_state['action']
						next_state = best_reachable_state['state']
						self.qmatrix[covered_state][action] = (
							self.qmatrix[covered_state][action]+
							self.alpha*(self.gamma*self.get_max_qvalue(next_state)-self.qmatrix[covered_state][action])
						)
			if self.epsilon_mode == EPSILON_MODES[1]:
				self.epsilon = max(self.epsilon_low, self.epsilon*self.epsilon_decay)
			self.curr_state = self.env.reset()


	def take_action(self, action):

		"""
		Effettua l'azione passata in ingresso

		Parameters
		-----------------------------------
		(int) action
			Azione che si vuole eseguire

		Returns
		-----------------------------------
		(int) done
			Indica se il nuovo stato è terminale
		"""

		self.curr_state, done = self.env.step(action)
		return done


	def get_action_from_qmatrix(self):

		"""
		Restituisce l'azione da intraprendere. La scelta puo' essere di tipo exploration
		oppure exploitation, ciò dipende dal valore di epsilon al momento dell'invocazione
		del metodo.

		Returns
		-----------------------------------
		(int) action
			Indica l'azione da intraprendere
		"""

		if (np.random.uniform(0, 1.0) > (1-self.epsilon)):
			return self.env.action_space.sample()
		potential_actions = []
		max_qvalue = np.max(self.qmatrix[self.curr_state])
		for action in range(0, self.env.action_space.n):
			if max_qvalue == self.qmatrix[self.curr_state][action]:
				potential_actions.append(action)
		return random.choice(potential_actions)


	def get_max_qvalue(self, state):

		"""
		Preleva il massimo valore Q dello stato passato in ingresso

		Parameters
		-----------------------------------
		(frozenbag) state
			Indica lo stato di cui si vuole conoscere il valore Q massimo

		Returns
		-----------------------------------
		(float) max
			Massimo valore Q in corrispondenza dello stato passato in ingresso
		"""

		return np.max(self.qmatrix[state])


	def get_best_next_reachable_states(self, state):

		"""
		Restituisce la lista degli stati ottimali immediatamente successivi allo stato
		passato in ingresso

		Parameters
		-----------------------------------
		(frozenbag) state
			Indica lo stato di cui si vogliono conoscere gli stati ottimali 
			immediatamente successivi

		Returns
		-----------------------------------
		(list) best_reachable_states
			Migliori stati immediatamente raggiungibili (in termini di Q) dallo 
			stato passato in ingresso
		"""

		best_reachable_states = []
		reachable_states = self.env.get_next_reachable_states(state)
		if len(reachable_states) != 0:
			for reachable_state in reachable_states:
				if len(best_reachable_states) == 0:
					best_reachable_states.append(reachable_state)
				else:
					if self.get_max_qvalue(best_reachable_states[0]['state']) <= self.get_max_qvalue(reachable_state['state']):
						if self.get_max_qvalue(best_reachable_states[0]['state']) < self.get_max_qvalue(reachable_state['state']):
							best_reachable_states = []
						best_reachable_states.append(reachable_state)
		return best_reachable_states


	def get_optimal(self):

		"""
		Restituisce la politica ottimale

		Returns
		-----------------------------------
		(frozenbag) optimal
			Politica ottimale appresa
		"""

		optimal = self.env.get_init_state()
		while not self.env.is_terminal_state(optimal):
			optimal = frozenbag(list(optimal) + [np.argmax(self.qmatrix[optimal])])
		return optimal