#!/bin/python2.7
import random
import collections
import matplotlib.pyplot as plt
import graph_tool as gt
from graph_tool import draw
import numpy as np
def distance(A, B):
    for i in range(min(len(A), len(B))):
        if A[i]!= B[i]:
            return len(A)+len(B)-2*i
    return len(A)-len(B)


#from graph_tool import draw
tree=gt.Graph()


def create_tree(tree,N):
    N_branch=100
    #start in the soma
    soma=tree.add_vertex()
    cent_order[soma]=0
    name[0]='0'
    tubuline[soma]=1
    ##create the tree
    leaves=collections.deque([soma])
    for i in range(N):
            #choose wheter the node will branch or not
        leave=random.choice(leaves)
        nameCount=0
        for target in tree.add_vertex(2):
            #create the edge and add to nodes' list
            tree.add_edge(leave,target)
            #assign property to the node
            name[target]=name[leave]+str(nameCount)
            nameCount=1
            cent_order[target]=cent_order[leave]+1
            tubuline[target]=0
            leaves.append(target)
    return tree,soma

def intranode_distance():
    #choose a random leaf
    print('process the graph')
    elected = soma
    while elected.out_degree()!=0:
        elected = random.choice( list(tree.vertices()))
    #measure distance for all the resting leaves
    for node in tree.vertices():
        if node.out_degree()==0:
            distanceNode[node]=distance(name[elected],name[node])
        else:
            distanceNode[node]=1
    #diffuse tubuline demand proportional respect the elected leaf distance
    tubuline.a=2**(0.01*distanceNode.a)
    cent_order.a=1000/(cent_order.a+1)
    size=draw.prop_to_size(cent_order)
    #plot it to understand
    pos=draw.sfdp_layout(tree)
    draw.graph_draw(tree,pos=pos,vertex_fill_color=tubuline,vertex_size=size)
    draw.graph_draw(tree,pos=pos,vertex_fill_color=tubuline,vertex_text=distanceNode)

#draw.graph_draw(tree)
def compute_prob(cent_order,tree,x):
    max_order=max(cent_order.get_array())
    #If T has a total of N nodes, the number of leaves is L = (N + 1)/2
    num_leaves=(tree.num_vertices()+1)/2
    mf_order = np.log2(num_leaves)
    #p=1-np.exp(np.log(x)/mf_order)
    return np.exp(-x*mf_order)

def branch_with_prob(tree,soma,x):
    children=soma.out_neighbours()
    while True:
        mynode=random.choice([next(children,soma),next(children,soma)])
        children=mynode.out_neighbours()
        if random.random()<compute_prob(cent_order,tree,x)*cent_order[mynode] and not mynode==soma:
   #         print (cent_order[mynode], name[mynode])
            oldedge=next(mynode.in_edges())
            oldfather=next(mynode.in_neighbours())
            tree.remove_edge(oldedge)
            newfather=tree.add_vertex()
            tree.add_edge(newfather,mynode)
            tree.add_edge(oldfather,newfather)
            tree.add_edge(newfather,tree.add_vertex())
            cent_order[newfather]=cent_order[mynode]
            name[newfather]=name[mynode]
            update_tree(newfather)
            tubuline[newfather]=10
            return cent_order[mynode]

def mydraw():
    pos=draw.sfdp_layout(tree)
    draw.graph_draw(tree,pos=pos,vertex_fill_color=tubuline)
    draw.graph_hist, bins = np.histogram(samples, bins=50)



def update_tree(newfather):
    nodes=collections.deque([newfather])
    while nodes:
        parent=nodes.pop()
        for enum, child in enumerate(parent.out_neighbours()):
            cent_order[child]=cent_order[parent]+1
            name[child]=name[parent]+str(enum)
            nodes.append(child)


def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

for x in frange(0.3,0.8,0.05):
    tree=gt.Graph()
    #the centrifugal order of the node
    cent_order=tree.new_vertex_property('int')
    #the binary name corresponding to tree's structure
    name=tree.new_vertex_property('string')
    #a competing resource
    tubuline=tree.new_vertex_property('float')
    #the topological distance froma certain node, to be computed
    distanceNode=tree.new_vertex_property('int')
    tree,soma=create_tree(tree,150)
    max_order=max(cent_order.get_array())
    num_leaves=(tree.num_vertices()+1)/2
    mf_order = np.log2(num_leaves)
    print("the graph is like this:\n")
    print("has ", tree.num_vertices(),"nodes")
    print("the max_order is", max_order)
    print("the mean_field order is", mf_order)
    print("the compute_prob is", compute_prob(cent_order,tree,x))
    samples=[]
    for i in range(100):
        samples.append(branch_with_prob(tree,soma,x))
    hist, bins = np.histogram(samples, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()



