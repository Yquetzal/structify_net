import math
import networkx as nx
import numpy as np
import scipy
#from scipy.stats import spearmanr
import pandas as pd
import structify_net as stn
#from structify_net.structureClasses import Rank_model, Graph_generator
import numbers
#from tqdm.auto import tqdm
import tqdm


def _largest_component(graph):
    Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
    G0 = graph.subgraph(Gcc[0])
    return G0

def has_giant_component(graph,threshold=0.9):
    """Check if the graph has a giant component
    
    Returns True if the ratio of the largest component to the total number of nodes is above the threshold

    Args:
        graph (nx.Graph): A graph
        threshold (float, optional): The threshold. Defaults to 0.9.

    Returns:
        _type_: True if the graph has a giant component
    """
    return giant_component_ratio(graph)>threshold
        

def giant_component_ratio(graph):
    """Ratio of the largest component to the total number of nodes

    Args:
        graph (nx.Graph): A graph

    Returns:
        _type_: _description_
    """
    nb_nodes=graph.number_of_nodes()
    largest_CC= len(max(nx.connected_components(graph), key=len))
    ratio_CC=largest_CC/nb_nodes
    return ratio_CC

def transitivity(graph):
    """Transitivity of the graph
    
    Args:
        graph (nx.Graph): A graph
    """
    return nx.transitivity(graph)

def average_clustering(graph):
    """Average clustering coefficient of the graph
    
    Args:
        graph (nx.Graph): A graph
    
    Returns:
        float: Average clustering coefficient, a number between 0 and 1
    """
    #print(graph.clustering())
    return np.average([cc for n,cc in  nx.clustering(graph).items()])