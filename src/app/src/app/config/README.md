{

	{

	"vcap": {
		"fps": 30                                    #editable
	},

	"highlighter": {
		"fps": 15,                                   #editable
		"res": "480p",                               #editable
		"format": "avi",                             #editable
		"duration": 10000,                           #editable
		"video_path": "/home/sysken",                #editable
		"video_name_prefix": "feedback_"             #editable
	},

	"analyzer": {
		"docker_image_repository": "affdex",         #editable
		"docker_image_tag": "4.0",                   #editable
		"csv_path": "/home/sysken"                   #editable
	},

	"win": {
		"win_title": "Mastermind RL",                #editable
		"default_theme": "D A R K   T H E M E",      #editable
		"geometry": "1200x820",                      #fixed
		"resizable": [false, false]                  #editable 
	},

	"rl": {
		"gym": "mastermind-v0",                      #fixed
		"max_evaluation": 3,                         #editable
		"min_evaluation": -1,                        #editable
		"no_actions": 4,                             #fixed
		"code_len": 3,                               #fixed
		"session_prefix": "session_",                #editable,
		"step_delay": 2000                           #editable        
	}

}

Apportare una modifica ai valori contrassegnati come #fixed potrebbe comportare errori
durante l'esecuzione dell'applicativo app.py