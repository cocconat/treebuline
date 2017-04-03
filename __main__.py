#!/bin/python2.7
import random
import collections
import matplotlib.pyplot as plt
import graph_tool as gt
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
    '''
    Create a random binary tree with N branching nodes
    '''
    #start in the soma
    tree=gt.Graph()
    cent_order=tree.new_vertex_property('int')
    soma=tree.add_vertex()
    cent_order[soma]=0
    ##create the tree with a simple deque algorithm
    leaves=collections.deque([soma])
    for i in range(N):
            #choose wheter the node will branch or not
        random.shuffle(leaves)
        leave=leaves.pop()
        for target in tree.add_vertex(2):
            #create the edge and add to nodes' list
            tree.add_edge(leave,target)
            cent_order[target]=cent_order[leave]+1
            leaves.append(target)
    return tree,cent_order

def branch_with_prob(tree,soma,x, func,cent_order):
    children=soma.out_neighbours()
    while True:
        mynode=random.choice([next(children,soma),next(children,soma)])
        children=mynode.out_neighbours()
        if random.random()<func(cent_order,tree,mynode,x) and not mynode==soma:
            return cent_order[mynode]

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


def run_parameter(A,B,dist_,intervals,size=300):
    '''
        Measure the resulting distribution for a set of values of x parameter
    '''
    # Get the tree and some other stuff
    tree,cent_order=create_tree(size)
    num_leaves=(tree.num_vertices()+1)/2
    max_order=max(cent_order.get_array())
    mf_order = np.log2(num_leaves)
    print("the graph is like this:\n")
    print("has ", tree.num_vertices(),"nodes")
    print("the max_order is", max_order)
    print("the mean_field order is", mf_order)
        # the tree's degree distribution
    norm_hist=np.histogram(cent_order.get_array(),range=(0,max_order+1), bins=max_order+1)[0]/float(len(cent_order.get_array()))
    print (norm_hist, sum(norm_hist))

    legend=[]
    c=[]

    #preapre the graphiscs
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
    #the topological distance froma certain node, to be computed
    for enum, x in enumerate(frange(A,B,(B-A)/intervals)):
        ## init the distribution, this save a lot of time! :)
        dist=init_distribution(x,tree,cent_order)
        start=time.clock()
        samples=[]
        soma=tree.vertex(0)

        ## run the algorithms
        for i in range(500):
            if dist_=="exp_normalization":
                samples.append(branch_with_prob(tree,soma,x,dist.exp_normalization,cent_order))
            if dist_=="inverse_exp":
                samples.append(branch_with_prob(tree,soma,x,dist.inverse_exp,cent_order))
            if dist_=="exponential":
                samples.append(branch_with_prob(tree,soma,x,dist.exponential,cent_order))
            if dist_=="R_algorithm":
                samples.append(R_algorithm(tree,soma,cent_order))

        ##plot the results
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

def run_scale_size(dist_,start ,end ,x ,intervals=10.):
    legend=[]
    c=[]
    ax1 = plt.subplot2grid((3,4), (0,0), colspan=3)
    ax2 = plt.subplot2grid((3,4), (1,0), colspan=3, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot2grid((3,4), (2,0), colspan=3)
    f = plt.figure()
    for enum, size in enumerate(np.logspace(start,end,intervals)):
        #create tree and propertymap
        tree,cent_order=create_tree(int(size))
        ##get properties
        max_order=max(cent_order.get_array())
        norm_hist=np.histogram(cent_order.get_array(),range=(0,max_order+1), bins=max_order+1)[0]/float(len(cent_order.get_array()))
        print (norm_hist, sum(norm_hist))
        ax3.set_ylabel("time for 50\n"
                        "iterations [s]")
        ax3.set_xlabel("parameter x")
        ax1.set_title ("Branching order for "+str(tree.num_vertices())+" nodes")
        ax2.set_xlabel("centrifugal order")
        ax1.set_ylabel("probability of occured \n"
                        "branches")
        dist=init_distribution(x,tree,cent_order)
        start=time.clock()
        samples=[]
        soma=tree.vertex(0)
        for i in range(50):
            if dist_=="exp_normalization":
                samples.append(branch_with_prob(tree,soma,x,dist.exp_normalization,cent_order))
            if dist_=="inverse_exp":
                samples.append(branch_with_prob(tree,soma,x,dist.inverse_exp,cent_order))
            if dist_=="exponential":
                samples.append(branch_with_prob(tree,soma,x,dist.exponential,cent_order))
            if dist_=="R_algorithm":
                samples.append(R_algorithm(tree,soma,cent_order))
        hist, bins = np.histogram(samples,range=(0,max_order+1), bins=max_order+1)
        hist=np.array(hist)/50./norm_hist
        width = 0.08
        center = (bins[:-1] + bins[1:]) / 2 +enum*width
        ax1.bar(center, hist, align='center', width=width)
        c.extend(ax2.plot(range(len(hist)),hist))
        ax2.scatter(range(len(hist)),hist)
        ax3.scatter(size,time.clock()-start)
        legend.append("size="+str(size)[:4])
    f.legend(c,legend,'right')
    #plt.tight_layout()
    #plt.savefig(dist_+"size_scale".png")
    plt.show()



#def run_scale_size

class init_distribution (object):
    def __init__(self,x,tree,cent_order):
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
            self.inv_exp_array_ = np.array([1-np.power(self.x,1./gamma) for gamma in range(1, self.max_order+2)])
            print ("the inv_exp_array is",self.inv_exp_array_)
        return self.inv_exp_array_

    def exp_normalization(self,cent_order,tree,node,x):
        return np.exp(-x*self.mf_order)*x*cent_order[node]

    def inverse_exp(self,cent_order,tree,node,x):
        array=self.inv_exp_array
        return array[cent_order[node]]

    def exponential(self,cent_order,tree,node,x):
        return np.exp(-x*self.mf_order)*np.power(2,cent_order[node])


def R_algorithm(tree,soma,cent_order):
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

##########MAIN###########
if sys.argv[1]=="scale":
    dist_=sys.argv[3]
    B=float(sys.argv[2])
    run_scale_size(dist_,start=2,end=3,x= B,intervals=10.)
else:
    A=float(sys.argv[1])
    B=float(sys.argv[2])
    dist_=sys.argv[3]
    run_parameter(A,B,dist_,intervals=10.)
