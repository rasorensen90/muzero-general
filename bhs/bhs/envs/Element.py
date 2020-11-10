# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:37:43 2019

@author: RTS
"""
class Element:
    def __init__(self, ID = None, inputElement = None, outputElement = None):
        self.ID = ID
        self.inputElements = [inputElement]
        self.outputElements = [outputElement]
        self.tote = None
        self.cost = [1]
    
    def setElements(self, inputElement, outputElement):
        self.setInputElements(inputElement)
        self.setOutputElements(outputElement)
    
    def setInputElements(self, connector, inputElement):
        self.inputElements[connector] = inputElement
    
    def setOutputElements(self, connector, outputElement):
        self.outputElements[connector] = outputElement
        
    def push(self, tote):
        if self.tote is None:
            self.tote = tote
            self.tote.element = self
        else:
            raise Exception('Target element was not empty')
    
    def pull(self):
        tote = self.tote
        self.tote = None
        return tote
    
    def isReadyToRecieve(self, elem=None):
        return self.tote == None
    
    def isToteReady(self):
        return self.tote is not None and not self.tote.moved

    def move(self, control = 0): # default control is 0
        if self.isToteReady():
#            self.tote.counter += 1
#            if self.tote.counter < 2:
#                self.tote.moved = True
#                return
            if self.outputElements[control].isReadyToRecieve(self):
                tote = self.pull()
                tote.moved = True
                
#                tote.counter = 0
                self.outputElements[control].push(tote)
            elif self.outputElements[control].tote is None or self.outputElements[control].tote.moved:
                self.tote.moved = True
                
#            print('Tote '+str(tote.ID)+' is moved from ' + str(self.ID) + ' to '+ str(self.outputElements[control].ID) + ' with control ' + str(control))

class Diverter(Element):
    def __init__(self, ID, inputElement = None, outputElement1 = None, outputElement2 = None): # output elements should be given as a list
        self.ID = ID
        self.inputElements = [inputElement]
        self.outputElements = [outputElement1, outputElement2]
        self.tote = None
        self.cost = [1,1]
        self.forced_control = None
            
class Merger(Element):
    def __init__(self, ID, inputElement1 = None, inputElement2 = None, outputElement = None): # input elements should be given as a list
        self.ID = ID
        self.inputElements = [inputElement1, inputElement2]
        self.outputElements = [outputElement]
        self.tote = None
        self.cost = [1]
        self.nextInputElementIdx = 0 #input 0 has first appearence priority
        
    def isReadyToRecieve(self, elem):
        if self.tote==None:
            if all([inputElem.isToteReady() for inputElem in self.inputElements]):
                if elem == self.inputElements[self.nextInputElementIdx]:
                    self.nextInputElementIdx = [1,0][self.nextInputElementIdx] # toggle nextInputElementIdx between 0 and 1
                    return True
                else:
                    return False
            else:
                self.nextInputElementIdx = 0 # reset self.nextInputElementIdx
                return True
        else:
            return False
        
class Toploader(Element):
    def __init__(self, ID, inputElement = None, outputElement = None):
        self.ID = ID
        self.inputElements = [inputElement]
        self.outputElements = [outputElement]
        self.totes = []
        self.tote = None
        self.cost = [1]
        
    def push(self, totes):
        if not isinstance(totes, list):
            totes = [totes]
        for t in totes:
            t.element = self
            self.totes.append(t)
        if self.tote is None:
            self.tote = self.totes.pop(0)
            for t in self.totes:
                t.moved=True 
    
    def pull(self):
        tote = self.tote
        if self.totes != []:
            self.tote = self.totes.pop(0)
            self.tote.moved = True
            for t in self.totes:
                t.moved=True 
        else:
            self.tote = None
        return tote
    
#    def isOccupied(self):
#        return False # Ulimited space
    
#    def isToteReady(self):
#        return self.totes != []
        
