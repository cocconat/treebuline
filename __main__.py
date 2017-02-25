import graph_tool as gt
import collections
import random
import numpy as np

#from graph_tool import draw
tree=gt.Graph()

soma=tree.add_vertex()

sources=collections.deque([soma])
cent_order=tree.new_vertex_property('int')
tubuline=tree.new_vertex_property('float')

cent_order[soma]=0
N_branch=100



while sources:
    node=sources.popleft()
    if N_branch>0:
        N_branch=N_branch-1
        if __debug__:
            print (N_branch)
        if random.choice([0,1,1,1,1]):
            for target in tree.add_vertex(2):
                    cent_order[target]=cent_order[node]+1
                    tree.add_edge(node,target)
                    sources.append(target)
    else:
        break

pos=gt.draw.sfdp_layout(tree)
size=gt.draw.prop_to_size(cent_order)
size.a=20/(size.a+2)
gt.draw.graph_draw(tree,pos=pos,vertex_size=size)


