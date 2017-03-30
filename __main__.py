#!/bin/python2.7
import random
import collections
import matplotlib.pyplot as plt
import graph_tool as gt
from graph_tool import draw
import numpy as np
import sys
import time
def distance(A, B):
    for i in range(min(len(A), len(B))):
        if A[i]!= B[i]:
            return len(A)+len(B)-2*i
    return len(A)-len(B)


#from graph_tool import draw


def create_tree(N):
    #start in the soma
    tree=gt.Graph()
    cent_order=tree.new_vertex_property('int')
    soma=tree.add_vertex()
    cent_order[soma]=0
    #name[soma]='0'
    #tubuline[soma]=1
    ##create the tree
    leaves=collections.deque([soma])
    for i in range(N):
            #choose wheter the node will branch or not
        random.shuffle(leaves)
        leave=leaves.pop()
        nameCount=0
        for target in tree.add_vertex(2):
            #create the edge and add to nodes' list
            tree.add_edge(leave,target)
            #assign property to the node
            #name[target]=name[leave]+str(nameCount)
            #nameCount=1
            cent_order[target]=cent_order[leave]+1
            #tubuline[target]=0
            leaves.append(target)
    return tree,cent_order

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


def branch_with_prob(tree,soma,x, func):
    children=soma.out_neighbours()
    while True:
        mynode=random.choice([next(children,soma),next(children,soma)])
        children=mynode.out_neighbours()
        if random.random()<func(cent_order,tree,mynode,x) and not mynode==soma:
   #         print (cent_order[mynode], name[mynode])
           # oldedge=next(mynode.in_edges())
           # oldfather=next(mynode.in_neighbours())
           # tree.remove_edge(oldedge)
           # newfather=tree.add_vertex()
           # tree.add_edge(newfather,mynode)
           # tree.add_edge(oldfather,newfather)
           # tree.add_edge(newfather,tree.add_vertex())
           # cent_order[newfather]=cent_order[mynode]
           # name[newfather]=name[mynode]
           # update_tree(newfather)
           # tubuline[newfather]=10
            return cent_order[mynode]

def draw_graph():
    pos=draw.sfdp_layout(tree)
    #draw.graph_draw(tree,pos=pos,vertex_fill_color=tubuline)
    #draw.graph_hist, bins = np.histogram(samples, bins=50)
    max_order=max(cent_order.get_array())
    colors = np.linspace(0, 1, max_order+1)
    c = tree.new_vertex_property("vector<double>")
    for v in tree.vertices():
        c[v] = [colors[cent_order[v]]]*3
    draw.graph_draw(tree, pos=pos, vertex_fill_color=c, output_size=(1000, 1000))

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

#draw.graph_draw(tree)
def prepare():
    print("the graph is like this:\n")
    print("has ", tree.num_vertices(),"nodes")
    print("the max_order is", max_order)
    print("the mean_field order is", mf_order)
    f = plt.figure()
    ax1 = plt.subplot2grid((3,4), (0,0), colspan=3)
    ax2 = plt.subplot2grid((3,4), (1,0), colspan=3, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot2grid((3,4), (2,0), colspan=3)
    ax3.set_ylabel("time for 500\n"
                    "iterations [s]")
    ax3.set_xlabel("parameter x")
    ax1.set_title ("Branching order for "+str(tree.num_vertices())+" nodes")
    ax2.set_xlabel("centrifugal order")
    ax1.set_ylabel("probability of occured \n"
                    "branches")
    norm_hist=np.histogram(cent_order.get_array(),range=(0,max_order+1), bins=max_order+1)[0]/float(len(cent_order.get_array()))
    print (norm_hist, sum(norm_hist))
    return ax1, ax2, ax3, f, norm_hist, max_order,mf_order,tree

def run_parameter(A,B,dist_):
    ax1, ax2, ax3, f, norm_hist, max_order,mf_order,tree = prepare()
    legend=[]
    c=[]
    #the centrifugal order of the node
    #the binary name corresponding to tree's structure
    #a competing resource
    #the topological distance froma certain node, to be computed
    for enum, x in enumerate(frange(A,B,(B-A)/10)):
        dist=init_distribution(x,tree)
        start=time.clock()
        samples=[]
        soma=tree.vertex(0)
        for i in range(500):
            if dist_=="exp_normalization":
                samples.append(branch_with_prob(tree,soma,x,dist.exp_normalization))
            if dist_=="inverse_exp":
                samples.append(branch_with_prob(tree,soma,x,dist.inverse_exp))
            if dist_=="exponential":
                samples.append(branch_with_prob(tree,soma,x,dist.exponential))
            if dist_=="R_algorithm":
                samples.append(R_algorithm(tree,soma))
        hist, bins = np.histogram(samples,range=(0,max_order+1), bins=max_order+1)
        hist=np.array(hist)/500./norm_hist
        width = 0.08
        center = (bins[:-1] + bins[1:]) / 2 +enum*width
        ax1.bar(center, hist, align='center', width=width)
        c.extend(ax2.plot(range(len(hist)),hist))
        ax2.scatter(range(len(hist)),hist)
        ax3.scatter(x,time.clock()-start)
        legend.append("x="+str(x)[:4])
    f.legend(c,legend,'right')
    plt.tight_layout()
    plt.savefig(dist_+str(A)+"to"+str(B)+".png")
    #plt.show()

#def run_scale_size(tree,dist_,start,end):


def run_scale_size

class init_distribution (object):
    def __init__(self,x,tree):
        self.x=x
        #If T has a total of N nodes, the number of leaves is L = (N + 1)/2
        self.mf_order=np.log2((tree.num_vertices()+1)/2)
        self.max_order=max(cent_order.get_array())
        self.mf_exp_=False
        self.stored=False
        self.inv_exp_array_=False

    @property
    def mf_exp(self):
        if not self.stored:
            self.stored=True
            self.mf_exp_=np.exp(-self.x*self.mf_order)
            print(self.mf_exp_)
        return self.mf_exp_

    @property
    def inv_exp_array(self):
        if not self.stored:
            self.stored=True
            self.inv_exp_array_ = np.array([1-np.power(self.x,-gamma) for gamma in range(0, self.max_order+1)])
            print ("the inv_exp_array is",self.inv_exp_array_)
        return self.inv_exp_array_

    def exp_normalization(self,cent_order,tree,node,x):
        return np.exp(-x*self.mf_order)*x*cent_order[node]

    def inverse_exp(self,cent_order,tree,node,x):
        array=self.inv_exp_array
        return array[cent_order[node]]

    def exponential(self,cent_order,tree,node,x):
        return np.exp(-x*self.mf_order)*np.power(2,cent_order[node])


def R_algorithm(tree,soma):
    next_node=soma
    select=soma
    for n in range(tree.num_vertices()):
        if (n==0 or 0==random.choice(range(n))):
            select=next_node
        try:
            next_node=random.choice(list(next_node.out_neighbours()))
        except:
            next_node=tree.vertex(random.choice(range(tree.num_vertices())))
    return cent_order[select]

tree,cent_order=create_tree(300)
num_leaves=(tree.num_vertices()+1)/2
max_order=max(cent_order.get_array())
mf_order = np.log2(num_leaves)

A=float(sys.argv[1])
B=float(sys.argv[2])
dist_=sys.argv[3]
run_parameter(A,B,dist_)
run_scale_size(dist_,start=100,end=1000)

