import itertools
import numpy as np
from scipy.special import comb
from bisect import bisect
import networkx as nx
from .viz import plot_rank_matrix

class Graph_generator:
    def __init__(self, sortedPairs, probas):
        self.sortedPairs=sortedPairs
        self.probas=probas
    
    def generate(self):
        return proba2graph(self.sortedPairs, self.probas)
    
    def ER(n, p):
        return proba2graph(list(itertools.combinations(range(n),2)), [p]*comb(n,2,exact=True))

class Rank_model:
    def __init__(self, sortedPairs, node_properties):
        self.sortedPairs=sortedPairs
        self.node_properties=node_properties
    
    def get_generator(self,epsilon,density=None,m=None):
        probas=rank2proba(self.sortedPairs,epsilon,density,m)
        return Graph_generator(self.sortedPairs, probas)

    def plot_matrix(self,nodeOrder=None,ax=None):
        plot_rank_matrix(self.sortedPairs,nodeOrder=nodeOrder,ax=ax)
        
def structure2graph(nodes,ranking_function,epsilon,density=None,m=None):
    """
    Return a graph from a structure function
    g: a networkx graph
    ranking_function: a function that takes a graph and returns a list of node pairs, ordered by the order in which they should appear in the graph
    epsilon: a weight between 0 and 1, that determines how much the graph should be structured
    """
    model = structure2model(nodes,ranking_function,epsilon,density,m)
    return model.generate()

def structure2model(nodes,ranking_function,epsilon,density=None,m=None):
    """
    Return a graph from a structure function
    g: a networkx graph
    ranking_function: a function that takes a graph and returns a list of node pairs, ordered by the order in which they should appear in the graph
    epsilon: a weight between 0 and 1, that determines how much the graph should be structured
    """
    if not isinstance(nodes, nx.Graph):
        g = nx.Graph()
        g.add_nodes_from(range(nodes))
        nodes=g

    sortedPairs=ranking_function(g)
    probas=rank2proba(sortedPairs,epsilon,density,m)
    return Graph_generator(sortedPairs, probas)

def proba2graph(sortedPairs, probas):
    """
    Return a graph from a list of probabilities
    probas: a list of probabilities, ordered by the order in which the edges should appear in the graph
    """
    edges=[]
    for i,p in enumerate(probas):
        if np.random.random()<p:
            edges.append(sortedPairs[i])
    all_nodes = set([n for e in sortedPairs for n in e])
    newG=nx.Graph()
    newG.add_nodes_from(all_nodes)
    newG.add_edges_from(edges)
    return newG

def rank2proba(sorted_pairs,epsilon,density=None,m=None,):
    if density==None:
        if m==None:
            raise Exception("You must specify either a density or a number of edges")
        else:
            density=m/len(sorted_pairs)
    
    det_function=relative_rank2proba_bezier(density,epsilon)
    probas=[]
    for i in range(len(sorted_pairs)):
        fraction_edges_in_bin = det_function(i / len(sorted_pairs))
        #nb_edges_in_bin=fraction_edges_in_bin*len(sorted_pairs)
        proba = fraction_edges_in_bin
        probas.append(proba)
    return probas

def relative_rank2proba_bezier(density, epsilon, nb_bins=100, plot=False):
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