import itertools
import numpy as np
from scipy.special import comb
from bisect import bisect
import networkx as nx
import structify_net as stn

# def _structure2graph(nodes,ranking_function,epsilon,density=None,m=None):
#     model = _structure2model(nodes,ranking_function,epsilon,density,m)
#     return model.generate()

# def _structure2model(nodes,ranking_function,epsilon,density=None,m=None):
#     if not isinstance(nodes, nx.Graph):
#         g = nx.Graph()
#         g.add_nodes_from(range(nodes))
#         nodes=g

#     sortedPairs=ranking_function(g)
#     probas=_rank2proba(sortedPairs,epsilon,density,m)
#     return Graph_generator(sortedPairs, probas)

def _proba2graph(sortedPairs, probas):
    edges=[]
    for i,p in enumerate(probas):
        if np.random.random()<p:
            edges.append(sortedPairs[i])
    all_nodes = set([n for e in sortedPairs for n in e])
    newG=nx.Graph()
    newG.add_nodes_from(all_nodes)
    newG.add_edges_from(edges)
    return newG

def _rank2proba(rank_model,epsilon,density=None,m=None,):

    sorted_pairs=rank_model.sortedPairs
    if density==None:
        if m==None:
            raise Exception("You must specify either a density or a number of edges")
        else:
            density=m/len(sorted_pairs)
    
    det_function=_relative_rank2proba_bezier(density,epsilon)
    probas=[]
    for i in range(len(sorted_pairs)):
        fraction_edges_in_bin = det_function(i / len(sorted_pairs))
        #nb_edges_in_bin=fraction_edges_in_bin*len(sorted_pairs)
        proba = fraction_edges_in_bin
        probas.append(proba)
    return probas

def _relative_rank2proba_bezier(density, epsilon, nb_bins=100, plot=False):
    if epsilon<0 or epsilon>1:
        raise Exception("weight must be between 0 and 1")
    # possible_edges=nb_nodes*(nb_nodes-1)/2
    if epsilon>0 and epsilon<1:
        structure_weight = np.log(0.5)/np.log(1-epsilon)
    if epsilon==1:
        structure_weight=0.0000001
    if epsilon==0:
        structure_weight=999999

    #print(structure_weight)
    points = [[0, 0], [density, density], [1, density]]
    weights = [1, structure_weight, 1]

    # Compute curve
    p_x, p_y = _bezier_curve(points, weights=weights, mesh_size=nb_bins)
    p_x = list(reversed(p_x))
    p_y = list(reversed(p_y))

    # Plot curve
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(p_x, p_y,'r-')
        ax1.set(ylabel="fraction of all possible edges encountered",xlabel="position of sorted node pairs")

        
        proba = [(p_y[i+1]-p_y[i])/(p_x[i+1]-p_x[i]) for i in range(len(p_y)-1)]+[0]
        ax2.plot(p_x, proba)
        ax2.set(ylabel="probability to observe an edge",xlabel="position of sorted node pairs")


        plt.show()

    derivatives = []
    for i in range(1, len(p_y)):
        derivatives.append((p_y[i]-p_y[i-1])/(p_x[i]-p_x[i-1]))


    derivatives.append(0)

    def to_return(x):
        position=bisect(p_x,x)-1
        return derivatives[position]

    return to_return



def _bernstein(i, n, t):
        """
        The i-th Bernstein polynomial of degree n
        """
        return comb(n, i) * (t ** (n - i)) * (1 - t) ** i

def _weighted_bezier_curve(nodes, weights, mesh_size=100):
    node_x = np.array([n[0] for n in nodes])
    node_y = np.array([n[1] for n in nodes])
    weights = np.array(weights)

    t = np.linspace(0.0, 1.0, mesh_size)
    weighted_bernstein = np.array(
        [_bernstein(i, len(nodes) - 1, t) * weights[i] for i in
         range(0, len(nodes))])

    sum_weighted_bernstein = np.sum(weighted_bernstein, axis=0)

    p_x = np.divide(np.dot(node_x, weighted_bernstein), sum_weighted_bernstein)
    p_y = np.divide(np.dot(node_y, weighted_bernstein), sum_weighted_bernstein)
    return p_x, p_y


def _bezier_curve(nodes, weights=None, mesh_size=100):
    """
        Returns the x- and y-arrays of points in the (weighted) Bézier curve
        constructed for the given nodes and weights.
        weights = array with length equal to number of nodes
        mesh_size = number of points in the Bézier curve rendering.
    """
    

    if weights is None:
        node_x = np.array([n[0] for n in nodes])
        node_y = np.array([n[1] for n in nodes])
        t = np.linspace(0.0, 1.0, mesh_size)
        numerator = np.array(
            [_bernstein(i, len(nodes) - 1, t) for i in
             range(0, len(nodes))])
        p_x = np.dot(node_x, numerator)
        p_y = np.dot(node_y, numerator)
        return p_x, p_y
    else:
        if mesh_size is None:
            return _weighted_bezier_curve(nodes, weights)
        else:
            return _weighted_bezier_curve(nodes, weights,
                                          mesh_size=mesh_size)