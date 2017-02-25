import graph_tool as gt
from graph_tool import draw
import collections
import random

def distance(A,B):
    for i in range(min(len(A),len(B))):
        if A[i]!=B[i]:
            return len(A)+len(B)-2*i
    return len(A)-len(B)


#from graph_tool import draw
tree=gt.Graph()
N_branch=50

#the centrifugal order of the node
cent_order=tree.new_vertex_property('int')
#the binary name corresponding to tree's structure
name=tree.new_vertex_property('string')
#a competing resource
tubuline=tree.new_vertex_property('float')
#the topological distance froma certain node, to be computed
distanceNode=tree.new_vertex_property('int')

#start in the soma
soma=tree.add_vertex()
cent_order[soma]=0
name[0]='0'

##create the tree
sources=collections.deque([soma])
while sources:
    node=sources.popleft()
    #stop in adding node after a certain number of branching events
    if N_branch>0:
        N_branch=N_branch-1
        #choose wheter the node will branch or not
        if random.choice([0,1,1,1,1]):
            nameCount=0
            for target in tree.add_vertex(2):
                #create the edge and add to nodes' list
                tree.add_edge(node,target)
                sources.append(target)
                #assign property to the node
                name[target]=name[node]+str(nameCount)
                nameCount=1
                cent_order[target]=cent_order[node]+1
    else:
        break
#choose a random leaf
print('process the graph')
elected=soma
while elected.out_degree()!=0:
    elected=random.choice(list(tree.vertices()))
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


