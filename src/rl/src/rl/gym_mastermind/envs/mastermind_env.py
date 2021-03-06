#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rl.rl_exceptions as rlexc
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from collections_extended import frozenbag
import itertools


# classes--------------------------------------------------------------------

class MastermindEnv(gym.Env):

	"""
	Rappresenta l'ambiente del gioco Mastermind

	Attributes
	-----------------------------------
	(list) secret
		Indica la sequenza che l'agente deve indovinare

	(Discrete) action_space
		Indica lo spazio delle azioni che l'agente puo' intraprendere

	(frozenbag) attempt
		Indica lo stato attuale dell'agente

	Methods
	-----------------------------------
	step(action)
		Attua i cambiamenti conseguenti dall'esecuzione di un'azione da parte
		di un'agente

	reset()
		Resetta l'ambiente al suo stato iniziale

	render()
		unsupported

	get_states()
		Restituisce tutti i possibili stati

	get_coverage(state)
		Restituisce la copertura (tutti gli stati che possono raggiungere in 
		modo diretto o indiretto) dello stato passato in ingresso

	get_next_reachable_states(state)
		Restituisce tutti i possibili stati immediatamente successivi 
		(raggiungibili con una sola azione) allo stato passato in ingresso

	get_init_state()
		Restituisce lo stato iniziale

	is_done()
		Verifica se l'agente si trova in uno stato terminale

	is_terminal_state(state)
		Verifica se lo stato passato in ingresso è terminale

	is_guessed(state)
		Verifica se lo stato passato in ingresso corrisponde
		alla sequenza segreta
	"""

	def __init__(self, no_pegs, secret):

		"""
		Parameters
		-----------------------------------
		(int) no_pegs 
			Indica il numero di pioli disponibili

		(list) secret
			Indica la sequenza che l'agente deve indovinare

		Raises
		-----------------------------------
		InvalidSecretError
			La lunghezza del codice segreto è 0

		InvalidActionError
			Nel codice è presente un'azione inesistente (piolo inesistente)
		"""

		super(MastermindEnv, self).__init__()
		if len(secret) == 0:
			raise rlexc.InvalidSecretError()
		else:
			for peg in secret:
				if not 0 <= peg <= no_pegs:
					raise rlexc.InvalidActionError(peg)

		self.action_space = spaces.Discrete(no_pegs)
		self.secret = secret
		self.attempt = frozenbag()


	def step(self, action):

		"""
		Attua i cambiamenti conseguenti dall'esecuzione di un'azione da parte
		di un'agente

		Parameters
		-----------------------------------
		(int) action
			Indica l'azione che l'agente vuole intraprendere

		Raises
		-----------------------------------
		InvalidActionError
			L'azione che l'agente vuole intraprendere è inesistente

		Returns
		-----------------------------------
		(frozenbag) attempt
			Nuovo stato dell'agente conseguente all'esecuzione dell'azione

		(bool) done
			Indica se il nuovo stato è terminale
		"""

		if not 0 <= action <= self.action_space.n:
			raise rlexc.InvalidActionError(action)
		self.attempt = frozenbag(list(self.attempt) + [action])
		return self.attempt, self.is_done()


	def reset(self):

		"""
		Resetta l'ambiente al suo stato iniziale

		Returns
		-----------------------------------
		(frozenbag) attempt
			Stato iniziale dell'agente
		"""

		self.attempt = frozenbag()
		return self.attempt


	def render(self, mode=None):

		"""
		unsopported
		"""

		raise rlexc.UnsupportedOperationError()


	def get_states(self):

		"""
		Restituisce tutti i possibili stati

		Returns
		-----------------------------------
		(list) states
			Lista degli stati
		"""

		states = []
		for k in range(len(self.secret)+1):
			for state in itertools.combinations_with_replacement(range(self.action_space.n), k): 
				states.append(frozenbag(state))
		return states


	def get_coverage(self, state):

		"""
		Restituisce la copertura (tutti gli stati che possono raggiungere in 
		modo diretto o indiretto) dello stato passato in ingresso

		Parameters
		-----------------------------------
		(frozenbag) state
			Indica lo stato di cui si vuole conoscere la copertura

		Raises
		-----------------------------------
		InvalidStateError
			Lo stato passato in ingresso non è valido (è inesistente)

		Returns
		-----------------------------------
		(set) coverage
			Copertura dello stato passato in ingresso
		"""

		if not state in self.get_states():
			raise rlexc.InvalidStateError(state)
		coverage = {state}
		for k in range(0, len(state)):
			for covered_state in itertools.combinations(state, k):
				coverage.add(frozenbag(covered_state))
		return coverage


	def get_next_reachable_states(self, state):

		"""
		Restituisce tutti i possibili stati immediatamente successivi 
		(raggiungibili con una sola azione) allo stato passato in ingresso

		Parameters
		-----------------------------------
		(frozenbag) state
			Indica lo stato di cui si vogliono conoscere gli stati 
			immediatamente raggiungibili

		Raises
		-----------------------------------
		InvalidStateError
			Lo stato passato in ingresso non è valido (è inesistente)

		Returns
		-----------------------------------
		(list) reachable_states
			Stati immediatamente raggiungibili
		"""

		if not state in self.get_states():
			raise rlexc.InvalidStateError(state)
		reachable_states = []
		if not self.is_terminal_state(state):
			for action in range(self.action_space.n):
				reachable_state = frozenbag(list(state) + [action])
				reachable_states.append({
					'action': action, 
					'state': reachable_state
				})
		return reachable_states


	def get_init_state(self):

		"""
		Restituisce lo stato iniziale

		Returns
		-----------------------------------
		(list) state
			Stato iniziale
		"""

		return frozenbag()


	def is_done(self):

		"""
		Verifica se l'agente si trova in uno stato terminale

		Returns
		-----------------------------------
		(bool) done
			Indica se l'agente si trova in uno stato terminale
		"""

		return self.is_terminal_state(self.attempt)


	def is_terminal_state(self, state):

		"""
		Verifica se lo stato passato in ingresso è terminale

		Parameters
		-----------------------------------
		(frozenbag) state

		Raises
		-----------------------------------
		InvalidStateError
			Lo stato passato in ingresso non è valido (è inesistente)

		Returns
		-----------------------------------
		(bool) terminal
			Indica se lo stato passato in ingresso è terminale
		"""

		if not state in self.get_states():
			raise rlexc.InvalidStateError(state)
		return len(self.secret) == len(state)


	def is_guessed(self):

		"""
		Verifica se la sequenza è stata indovinata

		Returns
		-----------------------------------
		(bool) guessed
			Indica se la sequenza segreta è stata indovinata
		"""
		return sorted(list(self.attempt)) == sorted(self.secret)