#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import collections
import utils_exceptions as uexc


# constants------------------------------------------------------------------

RESOLUTIONS = {
    '480p':  ( 640,  480),
    '720p':  (1280,  720),
    '1080p': (1920, 1080),
    '2k':    (2560, 1440),
    '4k':    (3840, 2160)
}

VIDEO_FORMATS = {
    'avi': cv2.VideoWriter_fourcc(*'MP42'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}


# classes--------------------------------------------------------------------

class FeedbackHighlighter:

    """
    Astrae una deque, utilizzata per la memorizzazione di frame, fornendo un unico
    metodo di inserimento (scroll)

    Attributes
    -----------------------------------
    (int) fps 
        Indica il numero di frame per secondo del video

    (str) resolution 
        Indica la risoluzione del video (ref. RESOLUTIONS)

    (str) video_format
        Indica il formato del video (ref. VIDEO_FORMATS)

    (str) duration
        Indica la durata del video in millisecondi

    (str) video_path
        Indica il path della cartella in cui salvare i video

    (deque) sliding_window_frames
        I frame passati alla funzione scroll vengono memorizzati in questa struttura dati. 
        La sua lunghezza massima è sempre dispari (in modo da avere una mediana perfetta)
        ed è data da:
        fps*duration/1000 if (fps*duration/1000)%2==1 else fps*duration/1000-1

    (deque) feedback_frames
        Tiene traccia dei frame indicati come frame di feedback e del corrispondente
        feedback_id. I suoi elementi sono coppie del tipo (frame, feedback_id)

    Methods
    -----------------------------------
    scroll(frame, is_feedback_frame=False, feedback_id=None)
        Inserisce il frame passato in ingresso nella deque sliding_window_frames ed 
        eventualmente anche nella deque feedback_frames se is_feedback_frame è True.
        Dopo aver effettuato le operazioni di inserimento verifica se il frame mediano
        della deque sliding_window_frames è un frame di feedback, in caso affermativo 
        salva il video (in tal modo i frame di feedback risulteranno sempre frame 
        centrali rispetto alla durata complessiva del video)

    save_video(feedback_id)
        Salva il video caratterizzato dai frame sliding_window_frames nella cartella
        indicata da video_path
    """

    def __init__(self, fps, resolution, video_format, duration, video_path):

        """
        Parameters
        -----------------------------------
        (int) fps 
            Indica il numero di frame per secondo del video

        (str) resolution 
            Indica la risoluzione del video (ref. RESOLUTIONS)

        (str) video_format
            Indica il formato del video (ref. VIDEO_FORMATS)

        (str) duration
            Indica la durata del video in millisecondi

        (str) video_path
            Indica il path della cartella in cui salvare i video

        Raises
        -----------------------------------
        InvalidResolutionError
            La risoluzione passata non è supportata (non è indicata in RESOLUTIONS)

        InvalidVideoFormatError
            Il formato video passato non è supportato (non è indicato in VIDEO_FORMAT)

        FileNotFoundError
            Il path della cartella in cui salvare i video è inesistente o non valido
        """

        if resolution not in RESOLUTIONS:
            raise uexc.InvalidResolutionError(resolution, RESOLUTIONS.keys())
        if video_format not in VIDEO_FORMATS:
            raise uexc.InvalidVideoFormatError(video_format, VIDEO_FORMATS.keys())
        if not os.path.isdir(video_path):
            raise uexc.FileNotFoundError(video_path)

        self.fps = fps
        self.resolution = resolution
        self.video_format = video_format
        self.duration = duration
        self.video_path = video_path.rstrip('/')
        sliding_window_maxlen = fps*duration/1000 if (fps*duration/1000)%2==1 else fps*duration/1000-1
        self.sliding_window_frames = collections.deque([None]*sliding_window_maxlen, maxlen=sliding_window_maxlen)
        self.feedback_frames = collections.deque()


    def scroll(self, frame, is_feedback_frame=False, feedback_id=None):

        """ 
        Inserisce il frame passato in ingresso nella deque sliding_window_frames ed 
        eventualmente anche nella deque feedback_frames se is_feedback_frame è True.
        Dopo aver effettuato le operazioni di inserimento verifica se il frame mediano
        della deque sliding_window_frames è un frame di feedback, in caso affermativo 
        salva il video (in tal modo i frame di feedback risulteranno sempre frame 
        centrali rispetto alla durata complessiva del video)

        Parameters
        -----------------------------------
        (numpy.ndarray) frame 
            Frame da salvare nella deque sliding_window_frames

        (bool) is_feedback_frame [opt, default = False]
            Indica se il frame passato in ingresso è stato catturato durante l'immisione
            di un feedback

        (str) feedback_id [opt, default = None]
            ID del feedback relativo a is_feedback_frame

        Raises
        -----------------------------------
        InvalidFeedbackIdError
            Il frame è stato indicato come frame di feedback ma il parametro feedback_id
            non è stato fornito

        VideoWriterInitializingError [propagated]
            Errore nella inizializzazione del Video Writer 

        Returns
        -----------------------------------
        (str) video_path
            Path dell'eventuale video salvato
        """

        if is_feedback_frame and feedback_id is None:
            raise uexc.InvalidFeedbackIdError()
        self.sliding_window_frames.append(frame)
        if is_feedback_frame:
            self.feedback_frames.append((frame, feedback_id))
        if (len(self.feedback_frames) > 0 and 
            (self.feedback_frames[0][0] == self.sliding_window_frames[int(len(self.sliding_window_frames)/2)]).all()):
            try:
                return self.save_video(self.feedback_frames.popleft()[1])
            except:
                raise
        return None


    def save_video(self, feedback_id):

        """ 
        Salva il video caratterizzato dai frame sliding_window_frames nella cartella
        indicata da video_path

        Parameters
        -----------------------------------
        (str) feedback_id [opt, default = None]
            ID del feedback relativo a is_feedback_frame

        Raises
        -----------------------------------
        VideoWriterInitializingError
            Errore nella inizializzazione del Video Writer

        Returns
        -----------------------------------
        (str) video_path
            Path del video salvato
        """

        video_path = self.video_path + '/' + feedback_id + '.' + self.video_format
        video_writer = cv2.VideoWriter(
            video_path, 
            VIDEO_FORMATS[self.video_format], 
            self.fps, 
            RESOLUTIONS[self.resolution]
        )
        if not video_writer.isOpened():
            raise uexc.VideoWriterInitializingError()
        for frame in self.sliding_window_frames:
            if frame is not None:
                video_writer.write(frame)
        return video_path