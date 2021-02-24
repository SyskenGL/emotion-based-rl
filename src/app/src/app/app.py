#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import rospkg
import Tkinter as tk
import numpy as np
import ttk
import tkFont
import cv2
import json
import time
import gym
import rl.gym_mastermind.envs.mastermind_env
from tinydb import TinyDB
from utils.feedback_highlighter import FeedbackHighlighter
from utils.emotion_analyzer import EmotionAnalyzer
from rl.agent import Agent
from threading import Thread, Condition
from collections import OrderedDict
from PIL import Image, ImageTk


# constants------------------------------------------------------------------

VCAP_SETTINGS = {
	'fps': 30
}

HIGHLIGHTER_SETTINGS = {
	'fps': 15,
	'res': '480p',
	'format': 'avi',
	'duration': 10000,
	'video_path': '/home/sysken',
	'video_name_prefix': 'feedback_'
}

ANALYZER_SETTINGS = {
	'docker_image_repository': 'affdex',
	'docker_image_tag': '4.0',
	'csv_path': '/home/sysken'
}

WIN_SETTINGS = {
	'win_title': 'Mastermind RL',
	'default_theme': 'D A R K   T H E M E',
	'geometry': '1200x820',
	'resizable': (False, False)
}

RL_SETTINGS = {
	'gym': 'mastermind-v0',
	'max_evaluation': 3,
	'min_evaluation': -1,
	'no_actions': 4,
	'code_len': 3
}


# application----------------------------------------------------------------

class Application:

	def __init__(self):

		root = tk.Tk()
		root.title(WIN_SETTINGS['win_title'])
		root.geometry(WIN_SETTINGS['geometry'])
		root.resizable(*WIN_SETTINGS['resizable'])
		root.protocol('WM_DELETE_WINDOW', self.destructor)

		self.widgets = {}
		self.widgets['root'] = root

		with open(rospkg.RosPack().get_path('app') + '/src/app/themes/themes.json') as themes:
  			self.themes_sheet = json.load(themes)

		self.fonts = None
		self.themes = None
		self.init_fonts()
		self.init_themes()
		self.curr_theme = WIN_SETTINGS['default_theme']

		self.stopped = True
		self.time_secs = 0
		self.last_time_secs = None
		self.rl_mainloop_thread = None
		self.exit = False

		self.rl_initialized = False
		self.rl_finished = False
		self.feedback_required = False
		self.feedback_provided = False
		self.reward = 0
		self.attempts = 0
		self.secret = np.full(RL_SETTINGS['code_len'], None)
		self.env = None
		self.agent = None

		self.init_gui()
		self.init_listeners()
		self.apply_theme()

		self.feedback_highlighter = FeedbackHighlighter(
			HIGHLIGHTER_SETTINGS['fps'], 
			HIGHLIGHTER_SETTINGS['res'],
			HIGHLIGHTER_SETTINGS['format'],
			HIGHLIGHTER_SETTINGS['duration'],
			HIGHLIGHTER_SETTINGS['video_path']
		)
		self.feedback_frame = False
		self.emotion_analyzer = EmotionAnalyzer(
			ANALYZER_SETTINGS['docker_image_repository'], 
			ANALYZER_SETTINGS['docker_image_tag'],
			HIGHLIGHTER_SETTINGS['video_path'],
			ANALYZER_SETTINGS['csv_path']
		)
		self.vcap = self.init_camera()
		self.webcam()

		self.rl_mainloop_cv = Condition()
		self.rl_mainloop()


	# customtk--------------------------------------------------

	def custom_label(self, master, x, y, height, width, *args, **kwargs):
	    frame = tk.Frame(master, height=height, width=width)
	    frame.pack_propagate(0)
	    frame.place(x=x, y=y)
	    label = tk.Label(frame, *args, **kwargs)
	    label.pack(fill=tk.BOTH, expand=1)
	    return label


	def custom_button(self, master, x, y, height, width, *args, **kwargs):
	    frame = tk.Frame(master, height=height, width=width)
	    frame.pack_propagate(0)
	    frame.place(x=x, y=y)
	    button = tk.Button(frame, *args, **kwargs)
	    button.pack(fill=tk.BOTH, expand=1)
	    return button


	def custom_option_menu(self, master, x, y, height, width, value, values, *args, **kwargs):
	    frame = tk.Frame(master, height=height, width=width)
	    frame.pack_propagate(0)
	    frame.place(x=x, y=y)
	    option_menu = tk.OptionMenu(frame, value, values, *args, **kwargs)
	    option_menu.pack(fill=tk.BOTH, expand=1)
	    return option_menu


	# init------------------------------------------------------

	def init_fonts(self):
		self.fonts = []
		for font in self.themes_sheet['fonts']:
			self.fonts.append(
				tkFont.Font(
					name=font['font_name'],
					family=font['font'], 
					size=font['font_size'], 
					weight=font['font_weight'],
			)
		)


	def init_themes(self):
		self.themes = {}
		for theme in self.themes_sheet['themes']:
			self.themes[theme['name']] = theme['widgets']


	def init_gui(self):

		# video preview

		video_preview_frame = tk.Frame(master=self.widgets['root'], width=640, height=510)
		video_preview_frame.place(x=40, y=40)
		video_preview_title = self.custom_label(video_preview_frame, 0, 0, 30, 640)
		video_preview_content = self.custom_label(video_preview_frame, 0, 30, 480, 640)
		self.widgets['video_preview_frame'] = video_preview_frame
		self.widgets['video_preview_title'] = video_preview_title
		self.widgets['video_preview_content'] = video_preview_content

		# attempts

		attempts_frame = tk.Frame(master=self.widgets['root'], width=200, height=110)
		attempts_frame.place(x=720, y=40)
		attempts_title = self.custom_label(attempts_frame, 0, 0, 30, 200)
		attempts_content = self.custom_label(attempts_frame, 0, 30, 80, 200)
		self.widgets['attempts_frame'] = attempts_frame
		self.widgets['attempts_title'] = attempts_title
		self.widgets['attempts_content'] = attempts_content

		# timer

		timer_frame = tk.Frame(master=self.widgets['root'], width=200, height=110)
		timer_frame.place(x=960, y=40)
		timer_title = self.custom_label(timer_frame, 0, 0, 30, 200)
		timer_content = self.custom_label(timer_frame, 0, 30, 80, 200)
		self.widgets['timer_frame'] = timer_frame
		self.widgets['timer_title'] = timer_title
		self.widgets['timer_content'] = timer_content

		# code selector

		code_selector_frame = tk.Frame(master=self.widgets['root'], width=440, height=360)
		code_selector_frame.place(x=720, y=190)
		code_selector_title = self.custom_label(code_selector_frame, 0, 0, 30, 440)
		code_selector_content = self.custom_label(code_selector_frame, 34, 54, 283, 373)
		code_selector_buttons = np.empty((RL_SETTINGS['code_len'], RL_SETTINGS['no_actions']), dtype=object)
		for step in range(RL_SETTINGS['code_len']):
			for action in range(RL_SETTINGS['no_actions']):
					padx = (0, 20) if action != RL_SETTINGS['no_actions']-1 else (0, 0)
					pady = (0, 24)
					tmp_frame = tk.Frame(code_selector_content, height=78, width=78)
					tmp_frame.pack_propagate(0)
					tmp_frame.grid(row=step, column=action, padx=padx, pady=pady)
					code_selector_buttons[step][action] = tk.Button(
						tmp_frame, 
						text=str(action), 
						command=None
					)
					code_selector_buttons[step][action].pack(fill=tk.BOTH, expand=1)
		self.widgets['code_selector_frame'] = code_selector_frame
		self.widgets['code_selector_title'] = code_selector_title
		self.widgets['code_selector_content'] = code_selector_content
		self.widgets['code_selector_buttons'] = code_selector_buttons

		# feedback evaluation

		feedback_evaluation_frame = tk.Frame(master=self.widgets['root'], width=640,height=110)
		feedback_evaluation_frame.place(x=40, y=590)
		feedback_evaluation_title = self.custom_label(feedback_evaluation_frame, 0, 0, 30, 640)
		feedback_evaluation_scale = tk.Scale(
			feedback_evaluation_frame,
			from_=RL_SETTINGS['min_evaluation'], 
			to=RL_SETTINGS['max_evaluation'], 
			length=480,
			resolution=0.1
		)
		feedback_evaluation_scale.place(x=40, y=40)
		feedback_evaluation_button = self.custom_button(feedback_evaluation_frame, 550, 42, 50, 50)
		self.widgets['feedback_evaluation_frame'] = feedback_evaluation_frame
		self.widgets['feedback_evaluation_title'] = feedback_evaluation_title
		self.widgets['feedback_evaluation_scale'] = feedback_evaluation_scale
		self.widgets['feedback_evaluation_button'] = feedback_evaluation_button

		# feedback indicator

		feedback_indicator_frame = tk.Frame(master=self.widgets['root'], width=200, height=110)
		feedback_indicator_frame.place(x=720, y=590)
		feedback_indicator_title = self.custom_label(feedback_indicator_frame, 0, 0, 30, 200)
		feedback_indicator_content = self.custom_label(feedback_indicator_frame, 0, 30, 80, 200)
		self.widgets['feedback_indicator_frame'] = feedback_indicator_frame
		self.widgets['feedback_indicator_title'] = feedback_indicator_title
		self.widgets['feedback_indicator_content'] = feedback_indicator_content
		
		# agent code

		code_frame = tk.Frame(master=self.widgets['root'], width=200, height=110)
		code_frame.place(x=960, y=590)
		code_title = self.custom_label(code_frame, 0, 0, 30, 200)
		code_content = self.custom_label(code_frame, 0, 30, 80, 200)
		self.widgets['code_frame'] = code_frame
		self.widgets['code_title'] = code_title
		self.widgets['code_content'] = code_content

		# control buttons

		flow_button = self.custom_button(self.widgets['root'], 40, 740, 40, 200)
		reset_button = self.custom_button(self.widgets['root'], 280, 740, 40, 200)
		self.widgets['flow_button'] = flow_button
		self.widgets['reset_button'] = reset_button

		# theme selector

		theme = tk.StringVar(self.widgets['root']) 
		theme.set(self.curr_theme)
		theme_selector = self.custom_option_menu(
			self.widgets['root'], 
			960, 740, 40, 200, 
			theme, 
			*self.themes.keys(),
			command=self.on_theme_changed
		)
		self.widgets['theme'] = theme
		self.widgets['theme_selector'] = theme_selector


	def init_camera(self):
		for camera in range(3):
			vcap = cv2.VideoCapture(camera)
			if vcap is not None and vcap.isOpened():
				return vcap
		return cv2.VideoCapture()


	def init_listeners(self):
		for step in range(RL_SETTINGS['code_len']):
			for action in range(RL_SETTINGS['no_actions']):
				self.widgets['code_selector_buttons'][step][action].configure(
					command=self.on_code_selector_button_clicked(step, action)
				)
		self.widgets['feedback_evaluation_button'].configure(
			command=self.on_feedback_evaluation_button_clicked
		)
		self.widgets['reset_button'].configure(command=self.on_reset_button_clicked)
		self.widgets['flow_button'].configure(command=self.on_flow_button_clicked)


	# gui-------------------------------------------------------

	def apply_theme(self):

		theme = self.themes[self.curr_theme]

		# root

		self.widgets['root'].configure(background=theme['root']['background'])

		# video preview

		self.widgets['video_preview_frame'].configure(
			bg=theme['video_preview_frame']['background']
		)
		self.widgets['video_preview_title'].configure(
			text = theme['video_preview_title']['text'],
			bg=theme['video_preview_title']['background'],
			fg=theme['video_preview_title']['foreground'],
			font=tkFont.Font(name=theme['video_preview_title']['font'], exists=True)
		)
		self.widgets['video_preview_content'].configure(
			bg=theme['video_preview_content']['background'],
			fg=theme['video_preview_content']['foreground'],
			font=tkFont.Font(name=theme['video_preview_content']['font'], exists=True)
		)

		# attempts

		self.widgets['attempts_frame'].configure(
			bg=theme['attempts_frame']['background']
		)
		self.widgets['attempts_title'].configure(
			text=theme['attempts_title']['text'],
			bg=theme['attempts_title']['background'],
			fg=theme['attempts_title']['foreground'],
			font=tkFont.Font(name=theme['attempts_title']['font'], exists=True)
		)
		self.widgets['attempts_content'].configure(
			bg=theme['attempts_content']['background'],
			fg=theme['attempts_content']['foreground'],
			font=tkFont.Font(name=theme['attempts_content']['font'], exists=True)
		)

		# timer

		self.widgets['timer_frame'].configure(
			bg=theme['timer_frame']['background']
		)
		self.widgets['timer_title'].configure(
			text=theme['timer_title']['text'],
			bg=theme['timer_title']['background'],
			fg=theme['timer_title']['foreground'],
			font=tkFont.Font(name=theme['timer_title']['font'], exists=True)
		)
		self.widgets['timer_content'].configure(
			bg=theme['timer_content']['background'],
			fg=theme['timer_content']['foreground'],
			font=tkFont.Font(name=theme['timer_content']['font'], exists=True)
		)

		# code selector

		self.widgets['code_selector_frame'].configure(
			bg=theme['code_selector_frame']['background']
		)
		self.widgets['code_selector_title'].configure(
			text=theme['code_selector_title']['text'],
			bg=theme['code_selector_title']['background'],
			fg=theme['code_selector_title']['foreground'],
			font=tkFont.Font(name=theme['code_selector_title']['font'], exists=True)
		)
		self.widgets['code_selector_content'].configure(
			bg=theme['code_selector_content']['background']
		)
		for step in range(RL_SETTINGS['code_len']):
			for action in range(RL_SETTINGS['no_actions']):
				self.widgets['code_selector_buttons'][step][action].configure(
					bg=theme['code_selector_button']['background'],
					fg=theme['code_selector_button']['foreground'],
					activebackground=theme['code_selector_button']['background_active'],
					activeforeground=theme['code_selector_button']['foreground_active'],
					disabledforeground=theme['code_selector_button']['foreground_disabled'],
					font=tkFont.Font(name=theme['code_selector_button']['font'], exists=True),
					highlightthickness=0, 
					bd=0
				)

		# feedback evaluation

		self.widgets['feedback_evaluation_frame'].configure(
			bg=theme['feedback_evaluation_frame']['background']
		)
		self.widgets['feedback_evaluation_title'].configure(
			text=theme['feedback_evaluation_title']['text'],
			bg=theme['feedback_evaluation_title']['background'],
			fg=theme['feedback_evaluation_title']['foreground'],
			font=tkFont.Font(name=theme['feedback_evaluation_title']['font'], exists=True)
		)
		self.widgets['feedback_evaluation_scale'].configure(
			tickinterval=1, 
			orient=tk.HORIZONTAL,
			bg=theme['feedback_evaluation_scale']['background'],
			fg=theme['feedback_evaluation_scale']['foreground'],
			troughcolor=theme['feedback_evaluation_scale']['trough'],
			font=tkFont.Font(name=theme['feedback_evaluation_scale']['font'], exists=True),
			highlightthickness=0,
			bd=0
		)
		self.widgets['feedback_evaluation_button'].configure(
			text=theme['feedback_evaluation_button']['text'],
			bg=theme['feedback_evaluation_button']['background'],
			fg=theme['feedback_evaluation_button']['foreground'],
			activebackground=theme['feedback_evaluation_button']['background_active'],
			activeforeground=theme['feedback_evaluation_button']['foreground_active'],
			disabledforeground=theme['feedback_evaluation_button']['foreground_disabled'],
			font=tkFont.Font(name=theme['feedback_evaluation_button']['font'], exists=True),
			highlightthickness=0, 
			bd=0,
		)

		# feedback indicator

		self.widgets['feedback_indicator_frame'].configure(
			bg=theme['feedback_indicator_frame']['background']
		)
		self.widgets['feedback_indicator_title'].configure(
			text=theme['feedback_indicator_title']['text'],
			bg=theme['feedback_indicator_title']['background'],
			fg=theme['feedback_indicator_title']['foreground'],
			font=tkFont.Font(name=theme['feedback_indicator_title']['font'], exists=True)
		)
		self.widgets['feedback_indicator_content'].configure(
			text=theme['feedback_indicator_content']['text'],
			bg=theme['feedback_indicator_content']['background'],
			fg=theme['feedback_indicator_content']['foreground'],
			font=tkFont.Font(name=theme['feedback_indicator_content']['font'], exists=True)
		)

		# agent code

		self.widgets['code_frame'].configure(
			bg=theme['code_frame']['background']
		)
		self.widgets['code_title'].configure(
			text=theme['code_title']['text'],
			bg=theme['code_title']['background'],
			fg=theme['code_title']['foreground'],
			font=tkFont.Font(name=theme['code_title']['font'], exists=True)
		)
		self.widgets['code_content'].configure(
			bg=theme['code_content']['background'],
			fg=theme['code_content']['foreground'],
			font=tkFont.Font(name=theme['code_content']['font'], exists=True)
		)

		# control buttons

		self.widgets['flow_button'].configure(
			bg=theme['flow_button']['background'],
			fg=theme['flow_button']['foreground'],
			activebackground=theme['flow_button']['background_active'],
			activeforeground=theme['flow_button']['foreground_active'],
			disabledforeground=theme['flow_button']['foreground_disabled'],
			font=tkFont.Font(name=theme['flow_button']['font'], exists=True),
			highlightthickness=0, 
			bd=0
		)
		self.widgets['reset_button'].configure(
			text=theme['reset_button']['text'],
			bg=theme['reset_button']['background'],
			fg=theme['reset_button']['foreground'],
			activebackground=theme['reset_button']['background_active'],
			activeforeground=theme['reset_button']['foreground_active'],
			disabledforeground=theme['reset_button']['foreground_disabled'],
			font=tkFont.Font(name=theme['reset_button']['font'], exists=True),
			highlightthickness=0, 
			bd=0
		)

		# theme selector

		self.widgets['theme_selector'].config(
			bg=theme['theme_selector']['background'],
			fg=theme['theme_selector']['foreground'],
			font=tkFont.Font(name=theme['theme_selector']['font'], exists=True),
			activebackground=theme['theme_selector']['background_active'],
			activeforeground=theme['theme_selector']['foreground_active'],
			highlightthickness=0,
			bd=0,
			relief=tk.FLAT,
			indicatoron=0,
			direction='above'
		)

		self.refresh()


	def refresh(self, refresh_type='all'):
		if refresh_type == 'all':
			self.update_timer()
			self.update_attempts()
			self.update_feedback_indicator()
			self.update_code()
			self.update_feedback_evaluation_scale()
			self.update_feedback_evaluation_button()
			self.update_flow_button()
			self.update_reset_button()
			self.update_code_selector()
		elif refresh_type == 'rl':
			self.update_attempts()
			self.update_code()
			self.update_feedback_indicator()
			self.update_feedback_evaluation_scale()
			self.update_feedback_evaluation_button()
			self.update_flow_button()
			self.update_reset_button()


	def update_attempts(self):
		attempts_str = str(self.attempts).replace('', ' ')[1: -1]
		self.widgets['attempts_content'].configure(text=attempts_str)


	def update_timer(self):
		mins, secs = divmod(int(round(self.time_secs)), 60)
		time_secs_str = str(mins).zfill(2) + ':' + str(secs).zfill(2)
		time_secs_str = time_secs_str.replace('', ' ')[1: -1]
		self.widgets['timer_content'].configure(text=time_secs_str)	


	def update_feedback_indicator(self):
		theme = self.themes[self.curr_theme]
		if self.feedback_required:
			fg = theme['feedback_indicator_content']['foreground_required']
		else:
			fg = theme['feedback_indicator_content']['foreground_not_required']
		self.widgets['feedback_indicator_content'].configure(fg=fg)


	def update_code(self):
		code = None
		if self.rl_initialized:
			code = list(self.agent.curr_state)
		if code is None or len(code) == 0:
			theme = self.themes[self.curr_theme]
			code_str = theme['code_content']['text_empty']
		else:
			code_str = '{' + str(code)[1:-1] + '}'
		self.widgets['code_content'].configure(text=code_str)


	def update_feedback_evaluation_scale(self):
		theme = self.themes[self.curr_theme]
		if self.feedback_required:
			state = tk.NORMAL
			troughcolor = theme['feedback_evaluation_scale']['trough']
		else:
			state = tk.DISABLED
			troughcolor = theme['feedback_evaluation_scale']['trough_disabled']
		self.widgets['feedback_evaluation_scale'].configure(state=state, troughcolor=troughcolor)


	def update_feedback_evaluation_button(self):
		if self.feedback_required:
			state = tk.NORMAL
		else:
			state = tk.DISABLED
		self.widgets['feedback_evaluation_button'].configure(state=state)


	def update_flow_button(self):
		theme = self.themes[self.curr_theme]
		if self.rl_finished:
			bg = theme['flow_button']['background_disabled']
			state = tk.DISABLED
			text = self.widgets['flow_button']['text']
		else:
			bg = theme['flow_button']['background']
			state = tk.NORMAL
			if self.stopped:
				text = theme['flow_button']['text_start']
			else:
				text = theme['flow_button']['text_stop']
		self.widgets['flow_button'].configure(state=state, bg=bg, text=text)


	def update_reset_button(self):
		theme = self.themes[self.curr_theme]
		if self.stopped:
			bg = theme['reset_button']['background']
			state = tk.NORMAL
		else:
			bg = theme['reset_button']['background_disabled']
			state = tk.DISABLED
		self.widgets['reset_button'].configure(state=state, bg=bg)


	def update_code_selector_button(self, step, action):
		theme = self.themes[self.curr_theme]
		if self.secret[step] is None:
			bg = theme['code_selector_button']['background']
			fg = theme['code_selector_button']['foreground']
			state = tk.NORMAL
		else:
			if action == self.secret[step]:
				bg = theme['code_selector_button']['background_selected']
				fg = theme['code_selector_button']['foreground_selected']
				if self.stopped:
					state = tk.NORMAL
				else:
					state = tk.DISABLED
			else:
				bg = theme['code_selector_button']['background_disabled']
				fg = theme['code_selector_button']['foreground']
				state = tk.DISABLED
		self.widgets['code_selector_buttons'][step][action].configure(bg=bg, fg=fg, state=state)


	def update_code_selector(self):
		for step in range(RL_SETTINGS['code_len']):
			for action in range(RL_SETTINGS['no_actions']):
				self.update_code_selector_button(step, action)


	def flash_code_selector_button(self, step, action, flash_bg_color, flash_count=3, delay=250):
		if flash_count > 0:
			self.widgets['code_selector_buttons'][step][action].configure(
					background=flash_bg_color
			)
			self.widgets['code_selector_buttons'][step][action].after(
				delay/2, 
				lambda: self.update_code_selector_button(step, action)
			)
			self.widgets['code_selector_buttons'][step][action].after(
				delay, 
				lambda: self.flash_code_selector_button(
					step, 
					action, 
					flash_bg_color,
					flash_count-1, 
					delay
			))


	def flash_error_code_selector(self):
		theme = self.themes[self.curr_theme]
		for step in range(RL_SETTINGS['code_len']):
			if self.secret[step] is None:
				for action in range(RL_SETTINGS['no_actions']):
					self.flash_code_selector_button(
						step, 
						action, 
						theme['code_selector_button']['flash_error']
					)


	def flash_guessed_code_selector(self):
		theme = self.themes[self.curr_theme]
		for step in range(RL_SETTINGS['code_len']):
			for action in range(RL_SETTINGS['no_actions']):
				self.flash_code_selector_button(
					step, 
					action, 
					theme['code_selector_button']['flash_guessed'],
					flash_count=1,
					delay=1500
				)


	def flash_action_code_selector(self, action):
		theme = self.themes[self.curr_theme]
		for step in range(RL_SETTINGS['code_len']):
			self.flash_code_selector_button(
				step, 
				action, 
				theme['code_selector_button']['flash_action'],
				flash_count=1,
				delay=1500
			)


	# listeners-------------------------------------------------

	def on_code_selector_button_clicked(self, step, action):
		def on_code_selector_button_clicked_listener():
			if self.secret[step] is None:
				self.secret[step] = action
			else:
				self.secret[step] = None
			self.update_code_selector()
		return on_code_selector_button_clicked_listener


	def on_flow_button_clicked(self):
		if self.stopped and (None in self.secret):
			self.flash_error_code_selector()
		else:
			with self.rl_mainloop_cv:
				self.stopped = not self.stopped
				self.rl_mainloop_cv.notifyAll()
			if not self.stopped:
				self.update_code_selector()
				self.timer()
			else:
				self.last_time_secs = None
		self.update_reset_button()
		self.update_flow_button()


	def on_reset_button_clicked(self):
		self.reset()
		self.refresh()


	def on_theme_changed(self, theme):
		self.curr_theme = theme
		self.apply_theme()


	def on_feedback_evaluation_button_clicked(self):
		self.reward = self.widgets['feedback_evaluation_scale'].get()
		with self.rl_mainloop_cv:
			self.feedback_required = False
			self.rl_mainloop_cv.notifyAll()
		self.feedback_provided = True
		self.feedback_frame = True
		self.update_feedback_indicator()
		self.update_feedback_evaluation_scale()
		self.update_feedback_evaluation_button()


	# status----------------------------------------------------

	def destructor(self):
		self.exit = True
		with self.rl_mainloop_cv:
			self.rl_mainloop_cv.notifyAll()
		self.widgets['root'].destroy()
		self.vcap.release()
		cv2.destroyAllWindows()  


	def reset(self):
		self.time_secs = 0
		self.last_time_secs = None
		self.rl_initialized = False
		self.rl_finished = False
		with self.rl_mainloop_cv:
			self.feedback_required = False
			self.stopped = True
			self.rl_mainloop_cv.notifyAll()
		self.feedback_provided = False
		self.feedback_frame = False
		self.reward = 0
		self.attempts = 0
		self.secret = np.full(RL_SETTINGS['code_len'], None)
		self.env = None
		self.agent = None


	# services--------------------------------------------------

	def webcam(self):
		success, frame = self.vcap.read()
		if success:	
			video_path = self.feedback_highlighter.scroll(
				cv2.flip(frame, 1), 
				self.feedback_frame, 
				HIGHLIGHTER_SETTINGS['video_name_prefix'] + str(int(time.time()))
			)
			self.feedback_frame = False
			if video_path is not None:
				Thread(target=lambda: self.emotion_analyzer.analyze(os.path.basename(video_path))).start()
			rgba_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGBA)
			img = Image.fromarray(rgba_frame)
			imgtk = ImageTk.PhotoImage(image=img)
			self.widgets['video_preview_content'].imgtk = imgtk
			self.widgets['video_preview_content'].configure(image=imgtk, text='')
		else:
			self.vcap.release()
			self.vcap = self.init_camera()
			theme = self.themes[self.curr_theme]
			self.widgets['video_preview_content'].configure(
				text=theme['video_preview_content']['text_error'],
				image=''
			)
		self.widgets['video_preview_content'].after(1000/VCAP_SETTINGS['fps'], self.webcam)


	def timer(self):
		if not self.stopped:
			self.last_time_secs = self.last_time_secs or time.time()
			now_time_secs = time.time()
			self.time_secs = (self.time_secs+now_time_secs-self.last_time_secs)%3600
			self.last_time_secs = now_time_secs
			self.update_timer()
			self.widgets['timer_content'].after(1000, self.timer)


	def rl_mainloop(self):

		def rl_mainloop_thread():

			while not self.exit:

				with self.rl_mainloop_cv:
					while self.stopped or self.feedback_required:
						if self.exit:
							return
						self.rl_mainloop_cv.wait()


				if not self.rl_initialized:

					self.env = gym.make(
						RL_SETTINGS['gym'], 
						no_pegs=RL_SETTINGS['no_actions'], 
						secret=self.secret
					)
					self.agent = Agent(self.env)
					self.rl_initialized = True

				else:

					if self.feedback_provided:
						self.agent.update_qmatrix(self.reward)
						self.feedback_provided = False
						self.agent.curr_state = self.env.reset()
					elif not self.feedback_required:
						action = self.agent.get_action_from_qmatrix()
						self.feedback_required = self.agent.take_action(action)
						if self.feedback_required:
							self.attempts += 1
							if self.env.is_guessed():
								self.stopped = True
								self.rl_finished = True
								self.feedback_required = False
								self.flash_guessed_code_selector()
							else:
								self.flash_action_code_selector(action)
						else:
							self.flash_action_code_selector(action)
					self.refresh('rl')
					time.sleep(1)
		

		self.rl_mainloop_thread = Thread(target=rl_mainloop_thread).start()

# main--------------------------------------------------------------------

if __name__ == "__main__":
	app = Application()
	tk.mainloop()