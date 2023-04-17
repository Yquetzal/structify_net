import itertools
import numpy as np
from scipy.special import comb
import networkx as nx
import structify_net.viz as viz
import structify_net as stn
import structify_net.transform as transform


class Graph_generator:
    """A graph generator
    
    This class instantiate graph generators. It is composed of a Rank Model Class, and a list of probabilites, such as the index of the probability in the list corresponds to the index of the edge in the rank model. 
    
    Attributes:
        rank_model (_type_): A rank model
        sortedPairs (_type_): the node pairs, sorted from the most likely to the least likely to be connected, same as in the rank model
        probas (_type_): the probabilities of observing an edge, one for each edge, sorted in the same probability as the sortedPairs
    """
    def __init__(self, rank_model, probas):
        """Create a graph generator

        Args:
            rank_model (_type_): A rank model
            probas (_type_): the probabilities of the edges in the rank model. Should be a list of the same length as the number of node pairs in the rank model
        """
        self.rank_model=rank_model
        self.sortedPairs=rank_model.sortedPairs
        self.probas=probas
    
    def generate(self):
        """Generate a graph based on this generator

        Returns:
            nx.Graph: a graph
        """
        return transform._proba2graph(self.sortedPairs, self.probas)
    
    def ER(n, p):
        """Generate an Erdos-Renyi graph

        Args:
            n (int): number of nodes
            p (float): probability of an edge

        Returns:
            nx.Graph: a graph
        """
        return transform._proba2graph(list(itertools.combinations(range(n),2)), [p]*comb(n,2,exact=True))
    
    def plot_proba_function(self,ax=None):
        """Plot the probability function
        This function plots the probability function of the graph generator. It is a plot of the probability of an edge to exist as a function of the rank of the edge in the rank model.

        Args:
            ax (_type_, optional): a matplotlib axis. Defaults to None.

        Returns:
            _type_: a matplotlib plot
        """
        return viz.plot_proba_function(self,ax=ax)
    
    def scores(self,scores=None,runs=1,details=False,latex_names=True):
        """return a score dataframe for this generator

        Args:
            scores (, optional): A list of scores to compute. Defaults to None.
            runs (int, optional): nombre de runs pour les scores. Defaults to 1.
            details (bool, optional): if True, return a dataframe with the details for each run, else return a dataframe with the mean. Defaults to False.
            latex_names (bool, optional): if True, use latex names for the scores. Defaults to True.

        Returns:
            pd.DataFrame: a dataframe with the scores
        """
        return stn.scores_for_generators(self,scores=scores,runs=runs,details=details,latex_names=latex_names)

class Rank_model:
    """A class to represent a rank model
    
    A rank model is composed of a list of node pairs, ordered such as the first ones are the most likely to be connected. It can also contain node properties, and a node order, which is the order in which the nodes should be plotted in a plot matrix. It can return a graph generator(provided a proper probability function), which can be used to generate graphs based on this rank model.
    
    Node properties are useful to match the edge order with the structure, it can be used to store the spatial position of nodes or the community membership of nodes, for example.
    
    Attributes:
        sortedPairs (_type_): A list of node pairs, ordered such as the first ones are the most likely to be connected
        node_properties (_type_, optional): node properties provided as networkx graph or dictionary. Defaults to None.
        node_order (_type_, optional): The order in which the nodes should be plotted in a plot matrix. Defaults to None.
    """
    def __init__(self, sortedPairs, node_properties=None,node_order=None):
        """ Initialize a rank model
        Args:
            sortedPairs (_type_):  A list of node pairs, ordered such as the first ones are the most likely to be connected
            node_properties (_type_, optional): node properties provided as networkx graph or dictionary. Defaults to None.
            node_order (_type_, optional): The order in which the nodes should be plotted in a plot matrix. Defaults to None.
        """
        self.sortedPairs=sortedPairs
        self.node_properties=node_properties
        self.node_order=node_order
        
    
    def get_generator(self,epsilon,density=None,m=None):
        """Return a graph generator for this rank model
        
        One of the two parameters density or m should be provided. If both are provided, m will be used.

        Args:
            epsilon (_type_): the epsilon parameter of the probability function
            density (_type_, optional): the density parameter of the probability function. Defaults to None.
            m (_type_, optional): the number of edges in the graph. Defaults to None.

        Returns:
            _type_: _description_
        """
        probas=stn.rank2proba(self,epsilon,density,m)
        return Graph_generator(self, probas)

    def plot_matrix(self,nodeOrder=None,ax=None):
        """Plot a matrix of the rank model
        
        Node pairs are ordered by rank. The color of the cell is the rank of the edge.

        Args:
            nodeOrder (_type_, optional): the order in which the nodes should be plotted. Defaults to None.
            ax (_type_, optional): an axis to plot on. Defaults to None.
        """
        if nodeOrder==None:
            nodeOrder=self.node_order
        viz.plot_rank_matrix(self,nodeOrder=nodeOrder,ax=ax)
        
    def generate_graph(self,epsilon,density=None,m=None):
        """Generate a graph from this rank model
        
        This function generates a graph from this rank model and a probability function. The probability function is defined by the epsilon parameter and the density parameter. One of the two parameters density or m should be provided. If both are provided, m will be used.

        Args:
            epsilon (_type_): the epsilon parameter of the probability function
            density (_type_, optional): the density parameter of the probability function. Defaults to None.
            m (_type_, optional): the number of edges in the graph. Defaults to None.

        Returns:
            _type_: _description_
        """
        return self.get_generator(epsilon,density,m).generate()
    
    def scores(self,m,scores=None,epsilons=0,runs=1,details=False,latex_names=True):
        """this function computes scores for a rank model
        
        It computes scores for a rank model, for a given number of edges m. The scores are computed for a range of epsilon values, and for a given number of runs.

        Args:
            m (_type_): number of edges in the graph
            scores (_type_, optional): the scores to compute. A dictionary with the score name as key and the score function as value. Defaults to None.
            epsilons (int, optional): either a single value or a list of values for epsilon. Defaults to 0.
            runs (int, optional): number of runs for each epsilon. Defaults to 1.
            details (bool, optional): If True, return a dataframe with the details for each run, else return a dataframe with the mean. Defaults to False.
            latex_names (bool, optional): Whether to use latex names for the scores. Defaults to True.

        Returns:
            _type_: _description_
        """
        return stn.scores_for_rank_models(self,m,scores=scores,epsilons=epsilons,runs=runs,details=details,latex_names=latex_names)
        