# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:37:43 2019

@author: RTS
"""
from Element import Element, Diverter, Merger, Toploader
import networkx as nx
import numpy as np
import scipy
#import pylab as plt
#from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
#import pygraphviz as pgv

def connect(element1=Element(), elem1conn = 0, element2=Element(), elem2conn=0, graph=nx.DiGraph()):
    element1.setOutputElements(elem1conn, element2)  
    element2.setInputElements(elem2conn, element1)
    graph.add_edge(element1.ID,element2.ID)
    
def add_straights(src, srcConnector, number, elems, graph):
    straights = []
    if elems != []:
        maxID = max([e.ID for e in elems])
    else:
        maxID = -1
    for i in range(number):
        straights.append(Element(ID=maxID+i+1))
    connect(src, srcConnector, straights[0], 0, graph)
    for i in range(len(straights)-1):
        connect(straights[i], 0,straights[i+1],0, graph)
    for s in straights:
        elems.append(s)
        
def createGCNMat(graph):
    adjMat = nx.to_numpy_matrix(graph)
    # myGraph = nx.convert_matrix.from_numpy_matrix(adjMat,create_using=nx.DiGraph())

    GCNMat = adjMat+np.identity(adjMat.shape[0])
    D_ = np.zeros_like(GCNMat)
    for i in range(GCNMat.shape[0]):
        D_[i,i] = np.sum(GCNMat[i,:])
    D_ = scipy.linalg.fractional_matrix_power(D_,-0.5)
        
    GCNMat = np.matmul(np.matmul(D_, GCNMat), D_)
    GCNMat = np.float32(GCNMat)
    
    return GCNMat

def env_0_0(): #16 elements
    graph = nx.DiGraph()
    
    elements = [Toploader(ID=0)]
    
    elements.append(Merger(ID=1))
    connect(elements[-2],0,elements[-1],0,graph)
    
    add_straights(src=elements[-1],srcConnector=0,number=6,elems=elements,graph=graph)
    
    elements.append(Diverter(ID=max([e.ID for e in elements])+1))
    connect(elements[-2],0,elements[-1],0,graph)
    
    add_straights(src=elements[-1],srcConnector=0,number=3,elems=elements,graph=graph)
    connect(elements[-1],0,elements[0],0,graph)
    
    add_straights(src=elements[8],srcConnector=1,number=4,elems=elements,graph=graph)
    connect(elements[-1],0,elements[1],1,graph)

    src = [0]
    dst = [3,7,9,14]
    
    GCNMat = createGCNMat(graph)
    print('Number of elements in environment: ', len(elements))
    return elements, dst, src, graph, GCNMat

def env_1_0(): #34 elements
    graph = nx.DiGraph()
    
    elements = [Toploader(ID=0)]
    P0 = elements[-1].ID
    
    elements.append(Element(ID=len(elements)))
    connect(elements[-2],0,elements[-1],0,graph)
    P1 = elements[-1].ID
    
    elements.append(Diverter(ID=len(elements)))
    connect(elements[-2],0,elements[-1],0,graph)
    div1 = elements[-1].ID
    
    add_straights(src=elements[-1],srcConnector=0,number=6,elems=elements,graph=graph)
    
    elements.append(Element(ID=len(elements)))
    connect(elements[-2],0,elements[-1],0,graph)
    P2 = elements[-1].ID
    
    add_straights(src=elements[-1],srcConnector=0,number=5,elems=elements,graph=graph)
    
    elements.append(Element(ID=len(elements)))
    connect(elements[-2],0,elements[-1],0,graph)
    P3 = elements[-1].ID
    
    add_straights(src=elements[-1],srcConnector=0,number=3,elems=elements,graph=graph)
    
    elements.append(Merger(ID=len(elements)))
    connect(elements[-2],0,elements[-1],1,graph)
    mer1 = elements[-1].ID
    
    elements.append(Element(ID=len(elements)))
    connect(elements[-2],0,elements[-1],0,graph)
    P4 = elements[-1].ID
    
    add_straights(src=elements[-1],srcConnector=0,number=4,elems=elements,graph=graph)
    connect(elements[-1],0,elements[P0],0,graph)
    
    add_straights(src=elements[div1],srcConnector=1,number=4,elems=elements,graph=graph)
    
    elements.append(Element(ID=len(elements)))
    connect(elements[-2],0,elements[-1],0,graph)
    P5 = elements[-1].ID
    
    add_straights(src=elements[-1],srcConnector=0,number=4,elems=elements,graph=graph)
    connect(elements[-1],0,elements[mer1],0,graph)
    
    src = [P0]
    dst = [P1, P2, P3, P4, P5]
    
    GCNMat = createGCNMat(graph)
    
    print('Number of elements in environment: ', len(elements))
    return elements, dst, src, graph, GCNMat


def env_2_0(): #101 elements
    elements = []
    src = []
    dst = []
    graph = nx.DiGraph()
    
    elements.append(Toploader(ID=0))
    P0 = elements[-1].ID
    src.append(P0)
    
    elements.append(Element(ID=len(elements)))
    P1 = elements[-1].ID
    connect(elements[P0],0,elements[P1],0,graph)
    dst.append(P1)
    
    elements.append(Diverter(ID=len(elements)))
    P1_o = elements[-1].ID
    connect(elements[P1],0,elements[P1_o],0,graph)
    
    add_straights(elements[P1_o],1,1,elements,graph)
    P1_o_1 = elements[-1].ID
    elements.append(Element(ID=len(elements)))
    P6 = elements[-1].ID
    connect(elements[P1_o_1],0,elements[P6],0,graph)
    dst.append(P6)
    
    elements.append(Diverter(ID=len(elements)))
    P6_o0 = elements[-1].ID
    connect(elements[P6],0,elements[P6_o0],0,graph)
    
    add_straights(elements[P6_o0],1,3,elements,graph)
    P6_o0_1 = elements[-1].ID
    
    elements.append(Merger(ID=len(elements)))
    P2_i0 = elements[-1].ID
    connect(elements[P6_o0_1],0,elements[P2_i0],1,graph)
    
    elements.append(Merger(ID=len(elements)))
    P2_i1 = elements[-1].ID
    connect(elements[P2_i0],0,elements[P2_i1],0,graph)
        
    elements.append(Merger(ID=len(elements)))
    P2_i2 = elements[-1].ID
    connect(elements[P2_i1],0,elements[P2_i2],0,graph)
        
    elements.append(Element(ID=len(elements)))
    P2 = elements[-1].ID
    connect(elements[P2_i2],0,elements[P2],0,graph)
    dst.append(P2)
    
    elements.append(Diverter(ID=len(elements)))
    P2_o = elements[-1].ID
    connect(elements[P2],0,elements[P2_o],0,graph)
        
    add_straights(elements[P2_o],1,1,elements,graph)
    P2_o_1 = elements[-1].ID
    elements.append(Element(ID=len(elements)))
    P7 = elements[-1].ID
    connect(elements[P2_o_1],0,elements[P7],0,graph)
    dst.append(P7)
    
    elements.append(Merger(ID=len(elements)))
    P3_i0 = elements[-1].ID
    connect(elements[P7],0,elements[P3_i0],1,graph)
    
    elements.append(Merger(ID=len(elements)))
    P3_i1 = elements[-1].ID
    connect(elements[P3_i0],0,elements[P3_i1],0,graph)    
    
    elements.append(Element(ID=len(elements)))
    P3 = elements[-1].ID
    connect(elements[P3_i1],0,elements[P3],0,graph)
    dst.append(P3)    
    
    elements.append(Diverter(ID=len(elements)))
    P3_o = elements[-1].ID
    connect(elements[P3],0,elements[P3_o],0,graph)
    
    add_straights(elements[P3_o],0,4,elements,graph)
    P3_o_0 = elements[-1].ID
    
    elements.append(Merger(ID=len(elements)))
    P4_i = elements[-1].ID
    connect(elements[P3_o_0],0,elements[P4_i],0,graph)
    
    elements.append(Element(ID=len(elements)))
    P4 = elements[-1].ID
    connect(elements[P4_i],0,elements[P4],0,graph)
    dst.append(P4)   
    
    elements.append(Diverter(ID=len(elements)))
    P4_o = elements[-1].ID
    connect(elements[P4],0,elements[P4_o],0,graph)    
    
    add_straights(elements[P4_o],0,8,elements,graph)
    P4_o_0 = elements[-1].ID
    
    elements.append(Merger(ID=len(elements)))
    P5_i = elements[-1].ID    
    connect(elements[P4_o_0],0,elements[P5_i],0,graph)    
    
    elements.append(Element(ID=len(elements)))
    P5 = elements[-1].ID
    connect(elements[P5_i],0,elements[P5],0,graph)
    dst.append(P5)
    
    elements.append(Diverter(ID=len(elements)))
    P5_o = elements[-1].ID
    connect(elements[P5],0,elements[P5_o],0,graph)
    
    add_straights(elements[P5_o],0,3,elements,graph)
    P5_o_0 = elements[-1].ID
    connect(elements[P5_o_0],0,elements[P0],0,graph)
    
    add_straights(elements[P1_o],0,5,elements,graph)
    P1_o_0 = elements[-1].ID
    connect(elements[P1_o_0],0,elements[P2_i0],0,graph)
    
    add_straights(elements[P2_o],0,2,elements,graph)
    P2_o_0 = elements[-1].ID
    connect(elements[P2_o_0],0,elements[P3_i0],0,graph)
            
    add_straights(elements[P3_o],1,10,elements,graph)
    P3_o_1 = elements[-1].ID
    connect(elements[P3_o_1],0,elements[P2_i2],1,graph)
    
    add_straights(elements[P4_o],1,14,elements,graph)
    P4_o_1 = elements[-1].ID
    connect(elements[P4_o_1],0,elements[P3_i1],1,graph)
    
    add_straights(elements[P5_o],1,17,elements,graph)
    P5_o_1 = elements[-1].ID
    connect(elements[P5_o_1],0,elements[P2_i1],1,graph)
    
    add_straights(elements[P6_o0],0,3,elements,graph)
    P6_o0_0 = elements[-1].ID
    
    elements.append(Diverter(ID=len(elements)))
    P6_o1 = elements[-1].ID
    connect(elements[P6_o0_0],0,elements[P6_o1],0,graph)
    
    add_straights(elements[P6_o1],1,5,elements,graph)
    P6_o1_1 = elements[-1].ID
    connect(elements[P6_o1_1],0,elements[P4_i],1,graph)
    
    add_straights(elements[P6_o1],0,3,elements,graph)
    P6_o1_0 = elements[-1].ID
    connect(elements[P6_o1_0],0,elements[P5_i],1,graph)
    
    
    GCNMat = createGCNMat(graph)
    [print(e.ID, e.__class__.__name__) for e in elements]
#    nx.draw_spectral(graph)
#    plt.show()
    print('Number of elements in environment: ', len(elements))
    return elements, dst, src, graph, GCNMat

def env_3_0(): #265 elements
    elements = []
    graph = nx.DiGraph()
    # P0>
    #    > P4 -
    # P1>

    elements.append(Toploader(ID=0))
    P0_i = elements[-1].ID
    
    add_straights(elements[P0_i],0,4,elements,graph)
    P0_o = elements[-1].ID
    
    elements.append(Merger(ID=len(elements)))
    P4 = elements[-1].ID
    connect(elements[P0_o],0,elements[P4],0,graph)
    
    
    elements.append(Diverter(ID=len(elements)))
    P5 = elements[-1].ID
    connect(elements[P4],0,elements[P5],0,graph)
        
    add_straights(elements[P5],1,24,elements,graph)
    P5_1 = elements[-1].ID
    
    elements.append(Merger(ID=len(elements)))
    P14_i = elements[-1].ID
    connect(elements[P5_1],0,elements[P14_i],0,graph)
    
    add_straights(elements[-1],0,29,elements,graph)
    P14 = elements[-1].ID
    
    elements.append(Merger(ID=len(elements)))
    P16 = elements[-1].ID
    connect(elements[P14],0,elements[P16],0,graph)
    
    elements.append(Diverter(ID=len(elements)))
    P17 = elements[-1].ID
    connect(elements[P16],0,elements[P17],0,graph)
    
    add_straights(elements[P17],1,9,elements,graph)
    P17_1 = elements[-1].ID
    
    elements.append(Element(ID=len(elements)))
    P20 = elements[-1].ID
    connect(elements[P17_1],0,elements[P20],0,graph)
    
    add_straights(elements[P20],0,4,elements,graph)
    P20_o = elements[-1].ID
    
    elements.append(Merger(ID=len(elements)))
    P22_i = elements[-1].ID
    connect(elements[P20_o],0,elements[P22_i],0,graph)
    
    add_straights(elements[P22_i],0,9,elements,graph)
    P22 = elements[-1].ID
    
    elements.append(Diverter(ID=len(elements)))
    P23 = elements[-1].ID
    connect(elements[P22],0,elements[P23],0,graph)
    
    elements.append(Diverter(ID=len(elements)))
    P24 = elements[-1].ID
    connect(elements[P23],0,elements[P24],0,graph)
    connect(elements[P24],0,elements[P0_i],0,graph)
    
    
    
    
    elements.append(Diverter(ID=len(elements)))
    P25 = elements[-1].ID
    connect(elements[P23],1,elements[P25],0,graph)
    
    elements.append(Toploader(ID=len(elements)))
    P2_i = elements[-1].ID
    connect(elements[P25],0,elements[P2_i],0,graph)
    
    add_straights(elements[P2_i],0,4,elements,graph)
    P2_o = elements[-1].ID
    
    elements.append(Merger(ID=len(elements)))
    P7 = elements[-1].ID
    connect(elements[P2_o],0,elements[P7],0,graph)
    
    elements.append(Diverter(ID=len(elements)))
    P8 = elements[-1].ID
    connect(elements[P7],0,elements[P8],0,graph)
    
    add_straights(elements[P8],1,24,elements,graph)
    P8_1 = elements[-1].ID
    
    elements.append(Merger(ID=len(elements)))
    P15_i = elements[-1].ID
    connect(elements[P8_1],0,elements[P15_i],0,graph)
    
    add_straights(elements[P15_i],0,29,elements,graph)
    P15 = elements[-1].ID
    
    elements.append(Merger(ID=len(elements)))
    P18 = elements[-1].ID
    connect(elements[P15],0,elements[P18],0,graph)

    elements.append(Diverter(ID=len(elements)))
    P19 = elements[-1].ID
    connect(elements[P18],0,elements[P19],0,graph)
    
    add_straights(elements[P19],1,9,elements,graph)
    P19_1 = elements[-1].ID
    
    elements.append(Element(ID=len(elements)))
    P21 = elements[-1].ID
    connect(elements[P19_1],0,elements[P21],0,graph)
    
    add_straights(elements[P21],0,4,elements,graph)
    P21_o = elements[-1].ID
    connect(elements[P21_o],0,elements[P22_i],1,graph)
    
    
    
    
    elements.append(Toploader(ID=len(elements)))
    P1_i = elements[-1].ID
    connect(elements[P24],1,elements[P1_i],0,graph)
    
    add_straights(elements[P1_i],0,4,elements,graph)
    P1_o = elements[-1].ID
    connect(elements[P1_o],0,elements[P4],1,graph)
    
    add_straights(elements[P5],0,9,elements,graph)
    P5_0 = elements[-1].ID
    
    elements.append(Merger(ID=len(elements)))
    P6 = elements[-1].ID
    connect(elements[P5_0],0,elements[P6],0,graph)
    
    elements.append(Diverter(ID=len(elements)))
    P9 = elements[-1].ID
    connect(elements[P6],0,elements[P9],0,graph)
    
    add_straights(elements[P9],0,9,elements,graph)
    P9_0 = elements[-1].ID
    
    elements.append(Merger(ID=len(elements)))
    P10 = elements[-1].ID
    connect(elements[P9_0],0,elements[P10],0,graph)
    
    elements.append(Diverter(ID=len(elements)))
    P11 = elements[-1].ID
    connect(elements[P10],0,elements[P11],0,graph)
    
    add_straights(elements[P11],1,5,elements,graph)
    P11_1 = elements[-1].ID
    connect(elements[P11_1],0,elements[P14_i],1,graph)
    
    

    
    elements.append(Toploader(ID=len(elements)))
    P3_i = elements[-1].ID
    connect(elements[P25],1,elements[P3_i],0,graph)
    
    add_straights(elements[P3_i],0,4,elements,graph)
    P3_o = elements[-1].ID
    connect(elements[P3_o],0,elements[P7],1,graph)
    
    add_straights(elements[P8],0,9,elements,graph)
    P8_0 = elements[-1].ID
    connect(elements[P8_0],0,elements[P6],1,graph)
    
    add_straights(elements[P9],1,9,elements,graph)
    P9_1 = elements[-1].ID   
    
    elements.append(Merger(ID=len(elements)))
    P12 = elements[-1].ID
    connect(elements[P9_1],0,elements[P12],0,graph)    
    
    elements.append(Diverter(ID=len(elements)))
    P13 = elements[-1].ID
    connect(elements[P12],0,elements[P13],0,graph)
    
    add_straights(elements[P13],1,5,elements,graph)
    P13_1 = elements[-1].ID
    connect(elements[P13_1],0,elements[P15_i],1,graph)
    
    
    
    
    add_straights(elements[P11],0,4,elements,graph)
    P11_0 = elements[-1].ID
    connect(elements[P11_0],0,elements[P12],1,graph)
    
    
    add_straights(elements[P13],0,4,elements,graph)
    P13_0 = elements[-1].ID
    connect(elements[P13_0],0,elements[P10],1,graph)

    
    add_straights(elements[P17],0,14,elements,graph)
    P17_0 = elements[-1].ID
    connect(elements[P17_0],0,elements[P18],1,graph)
    

    add_straights(elements[P19],0,14,elements,graph)
    P19_0 = elements[-1].ID
    connect(elements[P19_0],0,elements[P16],1,graph)  
    
    
    src = [P0_i,P1_i,P2_i,P3_i]
    dst = [P20,P21]
    
    for e in elements:
        print(e.ID, e.__class__.__name__) 
    GCNMat = createGCNMat(graph)
#    pos = nx.nx_pydot.graphviz_layout(graph, prog='dot')
#    nx.draw(graph,pos=pos)
#    plt.show()
    print('Number of elements in environment: ', len(elements))
    return elements, dst, src, graph, GCNMat


#elems = [Element(ID=i) for i in range(6)]
#elems[1] = Merger(ID=elems[1].ID)
#elems[2] = Diverter(ID=elems[2].ID)
#elems[5] = Toploader(ID=elems[5].ID)
##[print(e) for e in elems]
#
#connect(element1=elems[0], elem1conn=0, element2=elems[1], elem2conn=0)
#connect(element1=elems[1], elem1conn=0, element2=elems[2], elem2conn=0)
#connect(element1=elems[2], elem1conn=0, element2=elems[3], elem2conn=0)
#connect(element1=elems[2], elem1conn=1, element2=elems[4], elem2conn=0)
#connect(element1=elems[3], elem1conn=0, element2=elems[0], elem2conn=0)
#connect(element1=elems[4], elem1conn=0, element2=elems[1], elem2conn=1)
#connect(element1=elems[5], elem1conn=0, element2=elems[0], elem2conn=0)
#
#totes = []
#for i in range(5):
#    totes.append(Tote(i,dst=random.randint(0,5)))
##for t in totes:
##    elems[5].push(t)
#elems[5].push(totes)
##[print(e) for e in elems]
#for i in range(10):
#    print('')
#    step(elems,totes)
#    
#
#
