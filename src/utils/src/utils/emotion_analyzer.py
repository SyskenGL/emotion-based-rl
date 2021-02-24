#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import docker
import utils_exceptions as uexc


# constants------------------------------------------------------------------

FACE_MODES = {
    'large': 0,
    'small': 1
}


# classes--------------------------------------------------------------------


class EmotionAnalyzer:

    """
    Astrae un container docker per l'esecuzione di uno script affectiva il quale
    si occupa dell'individuazione delle emozioni presenti in un video

    Attributes
    -----------------------------------
    (docker.client.DockerClient) docker_client 
        Client docker

    (str) video_path 
        Indica il path condiviso (con il docker) che conterrà i video da analizzare

    (str) csv_path
        Indica il path in cui salvare il risultato dell'analisi del video

    (int) face_num
        Indica il numero di facce da analizzare

    (str) face_mode
        Indica il tipo di facce da analizzare (vicine oppure distanti)

    (bool) auto_remove
        Indica se il video deve essere eliminato al termine dell'analisi

    (docker.models.containers.Container) docker_container
        Container docker

    Methods
    -----------------------------------
    analyze_video(video_name)
        Analizza il video video_name se presente nella cartella indicata dal path video_path.
        Il risultato dell'analisi, ossia la serie di emozioni individuate nel tempo, è memorizzato 
        in un file csv avente lo stesso nome del video di cui si è richiesta l'analisi.

    """

    def __init__(self, docker_image_repository, docker_image_tag, video_path, csv_path, face_num=1, face_mode='large', auto_remove=True):

        """
        Parameters
        -----------------------------------
        (int) docker_image_repository 
            Repository dell'immagine AFFECTIVA

        (str) docker_image_tag 
            Tag dell'immagine AFFECTIVA

        (str) video_path
            Indica il path condiviso (con il docker) che conterrà i video da analizzare

        (str) csv_path
            Indica il path in cui salvare il risultato dell'analisi del video

        (str) face_num [opt, default = 1]
            Indica il numero di facce da analizzare

        (str) face_mode [opt, default = 'large']
            Indica il tipo di facce da analizzare (vicine oppure distanti)

        (str) auto_remove [opt, default = True]
            Indica se il video deve essere eliminato al termine dell'analisi

        Raises
        -----------------------------------
        FileNotFoundError
            I path video_path e/o csv_path sono inesistenti o non validi

        InvalidFaceNumError
            Il numero delle facce indicate è minore di 1

        InvalidFaceModeError
            Il modalità face_mode passata non è supportata (non è indicata in FACE_MODES)
        """

        if not os.path.isdir(video_path):
            raise uexc.FileNotFoundError(video_path)
        if not os.path.isdir(csv_path):
            raise uexc.FileNotFoundError(csv_path)
        if face_num < 0:
            raise uexc.InvalidFaceNumError(face_num)
        if face_mode not in FACE_MODES:
            raise uexc.InvalidFaceModeError(face_mode, FACE_MODES.keys())

        self.docker_client = docker.from_env()
        self.video_path = video_path.rstrip('/')
        self.csv_path = csv_path.rstrip('/')
        self.face_num = face_num
        self.face_mode = face_mode
        self.auto_remove = auto_remove       
        self.docker_container = self.docker_client.containers.create(
            image = docker_image_repository + ':' + docker_image_tag,
            volumes = {
                '/tmp/.X11-unix/': {
                    'bind': '/tmp/.X11-unix/', 
                    'mode': 'rw'
                }, 
                video_path: {
                    'bind': '/opt/testapp-artifact/video', 
                    'mode': 'rw'
                }
            },
            devices = [
                '/dev/video0:/dev/video0:rwm'
            ],
            stdin_open = True,
            privileged = True,
            auto_remove = True
        )

        self.docker_container.start()


    def __del__(self):

        """
        Ferma il container docker causandone la rimozione
        """

        try:
            self.docker_container.stop()
        except:
            pass
        

    def analyze(self, video_name):

        """ 
        Analizza il video video_name se presente nella cartella indicata dal path video_path.
        Il risultato dell'analisi, ossia la serie di emozioni individuate nel tempo, è memorizzato 
        in un file csv avente lo stesso nome del video di cui si è richiesta l'analisi.

        Parameters
        -----------------------------------
        (str) video_name
            Il nome del video da analizzare

        Raises
        -----------------------------------
        FileNotFoundError
            Il file video_name è inesistente o non valido

        Returns
        -----------------------------------
        (bool) error
            Indica se è stato riscontrato un errore durante l'analisi del video.
        (list) formatted_logs
            Logs registrati durante l'analisi del video
        """

        if not os.path.isfile(self.video_path + '/' + video_name):
            raise uexc.FileNotFoundError(self.video_path + '/' + video_name)

        command = (
            '/bin/bash -c \'cd testapp-artifact/build/video-demo/; ' + 
            './video-demo -d /opt/testapp-artifact/affdex-sdk/data/ -i /opt/testapp-artifact/video/{0} --draw 0 --numFaces {1} --faceMode {2}\''
        )
        exec_logs = self.docker_container.exec_run(
            command.format(video_name, self.face_num, FACE_MODES[self.face_mode]), 
            stdout = True, 
            stderr = True
        ) 

        if self.auto_remove:
            os.remove(self.video_path + '/' + video_name)
        csv_name = video_name.split(".", 1)[0] + ".csv"
        tmp_csv_path = self.video_path + '/' + csv_name
        csv_path = self.csv_path + '/' + csv_name
        if os.path.isfile(tmp_csv_path):
            error = False
            os.rename(tmp_csv_path, csv_path)
        else:
            error = True

        formatted_logs = []
        formatted_log = ""
        for log in exec_logs[1]:
            if log == '\n':
                formatted_logs.append(formatted_log)
                formatted_log = ""
            else:
                formatted_log = formatted_log + log

        return error, formatted_logs