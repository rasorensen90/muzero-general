# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:37:43 2019

@author: RTS
"""

class Tote:
    def __init__(self, ID, dst):
        self.ID = ID
        self.dst = dst
        self.element = None
        self.moved = False
        
        # TODO - Allow features more features than dst