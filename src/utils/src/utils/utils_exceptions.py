#!/usr/bin/env python
# -*- coding: utf-8 -*-


# exceptions-----------------------------------------------------------------

class InvalidResolutionError(ValueError):

    def __init__(self, resolution, resolutions, message='resolution must be one of '):
        self.resolution = resolution
        self.message = message + str(resolutions)
        super(ValueError, self).__init__(self.message)

    def __str__(self):
        return '\'{resolution}\' -> {message}'.format(resolution=self.resolution, message=self.message)



class InvalidVideoFormatError(ValueError):

    def __init__(self, video_format, video_formats, message='video_format must be one of '):
        self.video_format = video_format
        self.message = message + str(video_formats)
        super(ValueError, self).__init__(self.message)

    def __str__(self):
        return '\'{video_format}\' -> {message}'.format(video_format=self.video_format, message=self.message)



class InvalidFeedbackIdError(ValueError):

    def __init__(self, message='feedback_id must be not None'):
        self.message = message
        super(ValueError, self).__init__(self.message)



class VideoWriterInitializingError(Exception):
    
    def __init__(self, message='video writer initialization error'):
        self.message = message
        super(Exception, self).__init__(self.message)



class InvalidFaceNumError(ValueError):

    def __init__(self, face_num, message='face_num must be greater than or equal to 1'):
        self.face_num = face_num
        self.message = message
        super(ValueError, self).__init__(self.message)

    def __str__(self):
        return '\'{face_num}\' -> {message}'.format(face_num=self.face_num, message=self.message)



class InvalidFaceModeError(ValueError):

    def __init__(self, face_mode, face_modes, message='face_mode must be one of '):
        self.face_mode = face_mode
        self.message = message + str(face_modes)
        super(ValueError, self).__init__(self.message)

    def __str__(self):
        return '\'{face_mode}\' -> {message}'.format(face_mode=self.face_mode, message=self.message)



class FileNotFoundError(IOError):
    
    def __init__(self, file, message='file not found'):
        self.file = file
        self.message = message
        super(IOError, self).__init__(self.message)

    def __str__(self):
        return '\'{file}\' -> {message}'.format(file=self.file, message=self.message)