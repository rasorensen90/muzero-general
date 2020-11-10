# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:37:43 2019

@author: RTS
"""
import random
import numpy as np
from .envfactory_v4_1 import env_0_0, env_1_0, env_2_0, env_3_0
from .Element import Element, Diverter, Merger, Toploader
from .Tote import Tote
import gym
from gym import spaces
from .SPGraph import SPGraph, dijsktra

class BHSEnv(gym.Env):
    def __init__(self):
        
        self.elems, self.dst, self.src, self.graph, self.GCNMat = globals()["env_2_0"]()
        self.totes = []
        self.reward = 0
        self.action_space = []
        self.observation_space = []
        # self.numObsFeatures = 1
        self.stepnumber = 0
        self.steplimit = 200
        self.done = True
        self.default_rand = random.Random(0)
        self.rand_dst = random.Random(0)
        self.rand_src = random.Random(0)
        self.rand_numtotes = random.Random(0)
        self.randomize_numtotes = False#args.randomize_numtotes
        self.numtotes = 30#args.numtotes
        self.congestion = False
        self.congestion_counter = 0
        self.tote_info = {}
        self.deadlock = False
        self.diverters = [e for e in self.elems if isinstance(e, Diverter)]
        self.seed_ = None
        
        # if args.RL_diverters is not None:
        #     self.rl_diverter_ids = list(map(int,args.RL_diverters))
        # else:
        #     self.rl_diverter_ids = [e.ID for e in self.diverters]
        self.rl_diverter_ids = [e.ID for e in self.diverters]
        
        self.setSpaces()
        self.shortestPathTable = self.calcShortestPathTable()
        # print("RL_DIVERTERS",self.rl_diverter_ids)
        
    def updateObs(self): # onehot encoded
        self.obs = np.zeros(self.observation_space.shape)
        for e in self.elems:
            obs_e = np.zeros(len(self.src + self.dst))
            if e.tote is not None:
                idx = (self.src+self.dst).index(e.tote.dst)
                obs_e[idx] = 1
            self.obs[self.elems.index(e)] = obs_e
    
    def setDestination(self,tote):
        # TODO - auto detect destinations
        if tote.dst in self.src:
            tote.dst = self.rand_dst.choice(self.dst)#randint(1,len(self.elems)-1)
        else:
            tote.dst = self.rand_src.choice(self.src)
        if tote.ID not in self.tote_info:
            self.tote_info[tote.ID] = {'Tote': tote, 'TotalSteps': 0, 'Destinations': [], 'StepsPerDst': [], 'DstReached': [], 'StepNumber': []}
        self.tote_info[tote.ID]['Destinations'].append(tote.dst)
        self.tote_info[tote.ID]['StepNumber'].append(self.stepnumber)
        self.tote_info[tote.ID]['StepsPerDst'].append(0)
        self.tote_info[tote.ID]['DstReached'].append(0)
    
    def addTotes(self, numtotes):
        for i in range(numtotes):
            self.totes.append(Tote(i,dst=0))
            tote = self.totes[-1]
            self.tote_info[tote.ID] = {'Tote': tote, 'TotalSteps': 0, 'Destinations': [], 'StepsPerDst': [], 'DstReached': [], 'StepNumber': []}
            self.setDestination(tote)
            
            
        e_src = [e for e in self.elems if isinstance(e, Toploader)]
        for t in self.totes:
            src_ = self.rand_src.choice(e_src)
            src_.push(t)
    
    def setSpaces(self):
        # action space
        a = 2**len(self.diverters) # always two actions (This may change later)
       
        self.action_space = spaces.Discrete(a)
        
        # observation space
        obs_shape = (len(self.elems), len(self.src+self.dst))
        self.observation_space = spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.bool)
        self.obs = np.zeros(self.observation_space.shape)
        
    def checkForCongestion(self, dla=False, maxCongestion=1.0):
        current_congestion = 0
        for e in [e_ for e_ in self.elems if isinstance(e_, (Merger, Diverter, Toploader))]:
            dla_block = [0,0] # dla only works for diverters with two output directions
            for control in range(len(e.outputElements)):
                if not self.checkEdgeAvailability(e, control, maxCongestion):
                    e.cost[control] = 100
                    dla_block[control] = 1
                    self.congestion = True
                    current_congestion += 1
                else:
                    e.cost[control] = 1
                    dla_block[control] = 0
            if dla and isinstance(e, Diverter):
                if dla_block[0]:
                    e.forced_control = 1
                elif dla_block[1]:
                    e.forced_control = 0
                else: 
                    e.forced_control = None
                
        if self.congestion: # Can be from current step or previous step
            # update shortest path
            self.shortestPathTable = self.calcShortestPathTable()
            if current_congestion == 0: # If solved congestion 
                self.congestion=False

    def recursive_occupency(self, element, control):
        num_elements = 0
        num_occupied = 0
        
        if element.tote is not None and not element.tote.moved:
            num_elements += 1
            num_occupied += 1
        else:
            num_elements += 1
        
        if isinstance(element.outputElements[control], (Merger, Diverter, Toploader)):
            if element.outputElements[control].tote is not None and not element.outputElements[control].tote.moved:
                num_elements += 1
                num_occupied += 1
            else:
                num_elements += 1
            
            return num_elements, num_occupied
        
        else:
            num_elem, num_occ = self.recursive_occupency(element.outputElements[control], 0)
            num_elements += num_elem
            num_occupied += num_occ
            
            return num_elements, num_occupied
    
    def checkEdgeAvailability(self, element, control, maxCongestion = 1.0):
        num_elements, num_occupied = self.recursive_occupency(element,control)
        if num_occupied/num_elements < maxCongestion:   # Available
            return True
        else:                                           # Congested
            return False
        
        
    def calcShortestPathTable(self):        
        graph = SPGraph()
        [graph.add_node(e.ID) for e in self.elems]
        
        [[graph.add_edge(e.ID, e_out.ID, e.cost[e.outputElements.index(e_out)]) for e_out in e.outputElements] for e in self.elems]
        
        shortestPathTable = np.zeros([len(self.diverters),len(self.src+self.dst)])
        eID_src = [e.ID for e in self.diverters]
        eID_dst = self.src + self.dst
        
        eID = [e.ID for e in self.elems]
        
        for n_src in eID_src:
            _,path = dijsktra(graph,n_src)
            for n_dst in eID_dst:
                k=n_dst
                k_new = k
                while k_new != n_src:
                    k_new = path.get(k)
                    
                    if k_new != n_src:
                        k = k_new
                    if k == n_dst or k == None:
                        break
                
                e = self.elems[eID.index(n_src)]
                e_out = [e_out for e_out in e.outputElements if e_out.ID == k][0]
                ctrl = e.outputElements.index(e_out)

                shortestPathTable[eID_src.index(n_src),eID_dst.index(n_dst)] = ctrl
        return shortestPathTable
    
    def calcShortestPath(self, dynamic=False, dla=False, maxCongestion=1.0):
        if (dynamic):
            self.checkForCongestion(dla=dla, maxCongestion=maxCongestion)
        action = []
        eID_src = [e.ID for e in self.diverters]
        for e in self.diverters:
            
            if e.tote is not None:
                eID_dst = (self.src+self.dst).index(e.tote.dst)
            else:
                eID_dst = 0
            if e.forced_control is None:
                action.append(int(self.shortestPathTable[eID_src.index(e.ID)][eID_dst]))
            else:
                action.append(int(e.forced_control))
        return(action)
        
    def reset(self, total=True, seed=None, numtotes=None):
        if numtotes is not None:
            self.numtotes=numtotes
        if seed is None:
            if self.seed_ is None:
                seed_ = self.default_rand.choice(range(1000000))
            else: 
                seed_ = self.seed_
        else: 
            self.seed_ = seed
            seed_ = self.seed_
            
        self.rand_dst = random.Random(seed_)
        self.rand_src = random.Random(seed_+1)
        self.rand_numtotes = random.Random(seed_+2)
        if self.randomize_numtotes:
            self.numtotes = self.rand_numtotes.randint(1,len(self.elems))

        self.stepnumber = 0
        if total:
            self.tote_info = {}
            for e in self.elems:
                e.tote=None
                
                if isinstance(e, Toploader):
                     e.totes = []
                     e.tote = None
                   
            self.totes = []
            self.addTotes(self.numtotes)
        else:
            for key in self.tote_info:
                self.tote_info[key]['TotalSteps'] = 0
                self.tote_info[key]['Destinations'] = [t.dst for t in self.totes if t.ID == key]
                self.tote_info[key]['StepsPerDst'] = [0]
                self.tote_info[key]['DstReached'] = [0]
                self.tote_info[key]['StepNumber'] = [self.stepnumber]
        
        self.updateObs()
        self.congestion=False
        self.done = False
        return self.obs
    
    def findSPdiverters(self, load_based=False):
        if load_based:
            sp_diverters = []
            for d in self.diverters:
                ne0, no0 = self.recursive_occupency(d,0)
                ne1, no1 = self.recursive_occupency(d,1)
                if no0/ne1<=0.5 and no1/ne1<=0.5:
                    sp_diverters.append(d)
        else:
            sp_diverters = [div for div in self.diverters if div.ID not in self.rl_diverter_ids]
        
        return sp_diverters
    
    
    def step(self,action=[], shortestPath=False, dynamic=False, dla=False, maxCongestion=1.0): 
        reward = 0
        self.deadlock = False
        deadlock_= False
        sp_diverters = self.findSPdiverters(load_based=False)
        action_=[]
        if action is not []:
            action_ = np.array(list(format(action, '0'+str(len(self.diverters))+'b')), dtype=np.int) # convert from integer to binary array matching decisions at each diverter
            
        if shortestPath:
            action_ = self.calcShortestPath(dynamic=dynamic, dla=dla, maxCongestion=maxCongestion)
        elif self.rl_diverter_ids != None and sp_diverters != []:
            action_sp = self.calcShortestPath(dynamic=dynamic, dla=dla, maxCongestion=maxCongestion)
            indexes = [self.diverters.index(div) for div in sp_diverters]
            for i in indexes:
                action_[i] = action_sp[i]
        
        print(action_)
        e_ready = [e_ for e_ in self.elems if e_.tote is not None and not e_.tote.moved]
        e_old_1 = e_ready.copy()
        while e_ready != []:
            e_old_2 = e_ready.copy()
            for e in e_ready:
                if e in self.diverters:
                    print(self.diverters.index(e))
                    e.move(control=action_[self.diverters.index(e)])
                else:
                    e.move(control=0)
                if e.tote is None or e.tote.moved:
                    e_ready.remove(e)
            if e_ready == e_old_2:
                break
        if e_ready == e_old_1:
            print('Deadlock detected!')
            deadlock_ = True
            
        for t in self.totes:
            if t == t.element.tote: # ensures that tote is not in queue to toploader "outside" environment
                self.tote_info[t.ID]['TotalSteps']+=1
                self.tote_info[t.ID]['StepsPerDst'][-1]+=1
                self.tote_info[t.ID]['StepNumber'][-1]=self.stepnumber

            t.moved=False
            
            if t.dst == t.element.ID and t.dst in self.dst:
                reward += 1
                self.tote_info[t.ID]['DstReached'][-1]=1
                self.tote_info[t.ID]['Destinations'][-1] = t.element.ID # to store the used source
                self.setDestination(t)
            elif t.element.ID in self.src and t.dst in self.src:
                self.tote_info[t.ID]['DstReached'][-1]=1
                self.setDestination(t)
        self.updateObs()
        tote_info = self.tote_info.copy()
        if deadlock_:
            _ = self.reset(total=True)
            self.deadlock=True
            self.done = True
        self.stepnumber += 1
        if (self.stepnumber >= self.steplimit):
            self.done = True
        return self.obs, reward, self.done, tote_info#, action_

    def render(self, mode='human', close=False):
        pass
    
    def seed(self,seed):
        self.seed_=seed
        print("Seed set to: ", self.seed_)
        
        state = self.reset(total=True, seed=self.seed_) # Resetting env with new seed. Returning state
        return state

    def close(self):
        del self.env