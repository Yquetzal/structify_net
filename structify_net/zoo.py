import scipy
import scipy.spatial
import itertools
import random
import math
import networkx as nx
import numpy as np
import structify_net as stn

#We can choose the number of dimensions
#Add random positions in [0,1] in d dimensions, with names d1, d2, etc. to nodes
def _assign_ordinal_attributes(nb_nodes,d,g=None):
    if g==None:
        g=nx.Graph()
        g.add_nodes_from(range(nb_nodes))
    for i_dim in range(d):
        attributes= np.random.random(nb_nodes)
        nx.set_node_attributes(g,{i:a for i,a in enumerate(attributes)},"d"+str(i_dim+1))
    return(g)

def _n_to_graph(n):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    return g


def sort_ER(nodes):
    """Erdos-Renyi rank model
    
    Returns a rank model based on the Erdos-Renyi model. The Erdos-Renyi model is a random graph model where each pair of nodes is connected with probability p. For a rank model, all pairs of nodes have the same probability, so it simply shuffles the order of the nodes.

    Args:
        nodes (_type_): describe nodes of the graphs, either as a networkx graph (node names and node attributes are preserved) or an integer (number of nodes)

    Returns:
        :class:`structify_net.Rank_model`:: The corresponding rank model
    """
    if not isinstance(nodes,nx.Graph):
        g=_n_to_graph(nodes)
    else:
        g=nodes
    order = list(itertools.combinations(g.nodes,2))
    random.shuffle(order)
    return stn.Rank_model(order,g)
