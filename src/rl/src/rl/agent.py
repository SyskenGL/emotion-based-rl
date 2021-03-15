#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rl_exceptions as rlexc
import numpy as np
import itertools
import random
import math
from collections import OrderedDict
from collections_extended import frozenbag


# constants------------------------------------------------------------------

EXPLORATION_MODES = [
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

	(int) exploration_mode
		Indica quale strategia adottare nella selezione delle azioni da eseguire.
		Sono supportate le strategie e_greedy e e_decay

	(int) exploration_mode
		Indica quale strategia adottare nella selezione delle azioni da eseguire.
		Sono supportate le strategie e_greedy e e_decay

	(int) exploration_mode
		Indica quale strategia adottare nella selezione delle azioni da eseguire.
		Sono supportate le strategie e_greedy e e_decay

	(float) epsilon_decay
		Indica il fattore di decadimento di epsilon

	(float) epsilon_low
		Indica il valore minimo consentito di epsilon

	(list) td_history
		Lista dei TD riscontrati ad ogni aggiornamento

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
		Tutti gli stati terminali simili allo stato corrente vengono ricompensati in 
		proporzione alla somiglianza (1, 2, ..., n numeri uguali)

	shape_reward(state, reward, penalty)
		Modella la ricompensa moltiplicandola per 3 quando negativa e 
		aggiungendo una penalità proporzionale al numero di volte in cui 
		lo stato corrente è stato visitato -> R = reward - sqrt(x)/beta dove
		x è il numero delle volte in cui lo stato è stato visitato

	take_action(action)
		Effettua l'azione passata in ingresso

	get_action()
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

	def __init__(self, env, alpha=0.7, gamma=0.9, epsilon=0.999, beta=0.7, exploration_mode=EXPLORATION_MODES[1], epsilon_decay=0.9, epsilon_low=0.2):

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

		(float) beta [opt, default = 0.7]
			Indica il tasso di penalità

		(int) exploration_mode [opt, default = EXPLORATION_MODES[1]]
			Indica quale strategia di esplorazione

		(float) epsilon_decay [opt, default = 0.95]
			Indica il fattore di decadimento di epsilon

		(float) epsilon_low [opt, default = 0.35]
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
			La strategia indicata non è supportata (non è indicata in EXPLORATION_MODES)
		"""

		if not 0 <= alpha <= 1:
		    raise rlexc.InvalidAlphaError(alpha)
		if not 0 <= gamma <= 1:
		    raise rlexc.InvalidGammaError(gamma)
		if not 0 <= epsilon <= 1:
		    raise rlexc.InvalidEpsilonError(epsilon)
		if not exploration_mode in EXPLORATION_MODES:
		    raise rlexc.InvalidEpsilonModeError(exploration_mode, EXPLORATION_MODES)
		if not 0 <= epsilon_low <= 1:
		    raise rlexc.InvalidEpsilonError(epsilon)
		if not 0 <= beta:
			raise rlexc.InvalidBetaError(beta)

		self.env = env
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.beta = beta
		self.exploration_mode = exploration_mode
		self.epsilon_decay = epsilon_decay
		self.epsilon_low = epsilon_low
		self.curr_state = self.env.reset()
		self.qmatrix = self.init_qmatrix()

		
	def qmatrix_to_str(self):

		"""
		Stampa la matrice Q
		"""

		qmatrix_str = ''
		for state in self.qmatrix.keys():
			qmatrix_str += ('   {0:>{1}}: '.format('{' + str(list(state))[1:-1] + '}', 10)
				+ '\n\t\tqvalues              -> ' + str(self.qmatrix[state]['qvalues'])
				+ '\n\t\ttd_errors            -> ' + str(self.qmatrix[state]['td_errors'])
				+ '\n\t\ttd_errors_variations -> ' + str(self.qmatrix[state]['td_errors_variations'])
				+ '\n\t\tvisits               -> ' + str(self.qmatrix[state]['visits']) + '\n\n'
			)
		return qmatrix_str


	def init_qmatrix(self):
		
		"""
		Inizializza la matrice Q
		"""

		qmatrix = OrderedDict()
		for state in self.env.get_states():
		    qmatrix[state] = {
		    	'qvalues': np.full(self.env.action_space.n, 0, dtype=float),
		    	'td_errors': np.full(self.env.action_space.n, 0, dtype=float),
		    	'td_errors_variations': np.full(self.env.action_space.n, 0, dtype=float),
		    	'visits': 0
		    }
		return qmatrix


	def update_qmatrix(self, reward):

		"""
		Aggiorna la matrice Q effettuando una propagazione della ricompensa all'indietro.
		Tutti gli stati terminali simili allo stato corrente vengono ricompensati in 
		proporzione alla somiglianza (1, 2, ..., n numeri uguali)

		Parameters
		-----------------------------------
		(float) reward
			Indica la ricompensa che l'utente ha assegnato allo stato (finale) raggiunto
		"""

		def update(state, reward):

			coverage = self.env.get_coverage(state)
			for covered_state in sorted(list(coverage), key=len, reverse=True):
				if not self.env.is_terminal_state(covered_state):
					for reachable_state in self.env.get_next_reachable_states(covered_state):
						if reachable_state['state'] in coverage:
							action = reachable_state['action']
							td = self.alpha*self.gamma*self.get_max_qvalue(reachable_state['state'])-self.qmatrix[covered_state]['qvalues'][action]
							self.qmatrix[covered_state]['qvalues'][action] = self.qmatrix[covered_state]['qvalues'][action]+td
							self.qmatrix[covered_state]['td_errors_variations'][action] = abs(self.qmatrix[covered_state]['td_errors'][action]-abs(td))
							self.qmatrix[covered_state]['td_errors'][action] = abs(td)
				else:
					td = self.alpha*reward-self.qmatrix[covered_state]['qvalues'][0]
					self.qmatrix[covered_state]['qvalues'][:] = self.qmatrix[covered_state]['qvalues'][:]+td
					self.qmatrix[covered_state]['td_errors_variations'][:] = abs(self.qmatrix[covered_state]['td_errors'][0]-abs(td))
					self.qmatrix[covered_state]['td_errors'][:] = abs(td)

		if self.env.is_terminal_state(self.curr_state):
			reward = self.shape_reward(self.curr_state, reward, False)
			for state in self.qmatrix.keys():
				common_elements = len(self.curr_state&state)
				if self.env.is_terminal_state(state) and self.qmatrix[state]['visits'] == 0 and (1 <= common_elements <= 2):
					update(state, reward/self.env.get_terminal_state_len()*common_elements)
			update(self.curr_state, self.shape_reward(self.curr_state, reward, True))
			if self.exploration_mode == EXPLORATION_MODES[1]:
				self.epsilon = max(self.epsilon_low, self.epsilon*self.epsilon_decay)

		for covered_state in self.env.get_coverage(self.curr_state):
			self.qmatrix[covered_state]['visits'] += 1


	def shape_reward(self, state, reward, penalty):

		"""
		Modella la ricompensa moltiplicandola per 3 quando negativa e 
		aggiungendo una penalità proporzionale al numero di volte in cui 
		lo stato corrente è stato visitato -> R = reward - sqrt(x)/beta dove
		x è il numero delle volte in cui lo stato è stato visitato

		Parameters
		-----------------------------------
		(frozenbag) state
			Stato a cui è attribuita la ricompensa

		(float) reward
			Indica la ricompensa che l'utente ha assegnato allo stato raggiunto

		(boolean) penalty
			Indica se la penalità deve essere applicata

		Returns
		-----------------------------------
		(float) reward
			Ricompensa modellata
		"""

		reward = reward if reward >= 0 else reward*3
		if penalty:
			penalty = -(math.sqrt(self.qmatrix[self.curr_state]['visits'])/self.beta)
			reward += penalty
		return reward


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


	def get_action(self):

		"""
		Restituisce l'azione da intraprendere.

		Returns
		-----------------------------------
		(int) action
			Indica l'azione da intraprendere
		"""

		if (np.random.uniform(0, 1.0) > (1-self.epsilon)):
			return self.env.action_space.sample()
		potential_actions = []
		max_qvalue = self.get_max_qvalue(self.curr_state)
		for action in range(0, self.env.action_space.n):
			if max_qvalue == self.qmatrix[self.curr_state]['qvalues'][action]:
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

		return np.max(self.qmatrix[state]['qvalues'])


	def get_argmax_action(self, state):

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

		return np.argmax(self.qmatrix[state]['qvalues'])


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
			optimal = frozenbag(list(optimal) + [self.get_argmax_action(optimal)])
		return optimal