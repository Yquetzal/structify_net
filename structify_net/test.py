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

def average_shortest_path_length(graph,normalized=True):
    """The average shortest path length of the graph
    
    If the graph has a giant component, the average shortest path length is computed on the giant component.
    If normalized is true, compute a normalized version, defined between 0 and 1, where 1 is the shortest possible path length and 0 is the longest possible path length. It is computed as 1/(normalized_shortest+1), where normalized_shortest is the normalized shortest path length, defined as max(0,(graph_shortest-2)).

    Args:
        graph (_type_): _description_
        normalized (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if has_giant_component(graph):
        if giant_component_ratio(graph)<1:
            graph=_largest_component(graph)
        
        graph_shortest=nx.average_shortest_path_length(graph)
        
       
        if normalized:
            n= graph.number_of_nodes()
            #avg_degree=graph.number_of_edges()/n*2
            #ref_shortest = (np.log(n)-0.57722)/np.log(avg_degree)+0.5
            #normalized_shortest = max(0,(graph_shortest-2))/(ref_shortest-2)
            #print(normalized_shortest,graph_shortest,ref_shortest)
            normalized_shortest = max(0,(graph_shortest-2))
            return 1/(normalized_shortest+1)
        else:
            return graph_shortest
             
    #         ref = Graph_generator.ER(graph.number_of_nodes(),graph.number_of_edges()/graph.number_of_nodes()**2)
    #         #ref_shortest = nx.average_shortest_path_length(ref)
    #         n= graph.number_of_nodes()
    #         avg_degree=graph.number_of_edges()/n
    #         ref_shortest = (np.log(n)-0.57722)/np.log(avg_degree)+0.5
            
    #         max_diameter = math.log2(graph.number_of_nodes()) / math.log2(graph.number_of_nodes()/graph.number_of_edges())
    #         max_avg_distance = max_diameter / 2
    #         print(ref_shortest,max_avg_distance,graph_shortest)
    #         min_value=1
    #         if graph_shortest<ref_shortest:
    #             graph_shortest=(graph_shortest-min_value)/(ref_shortest-min_value)/2
    #         else:
    #             graph_shortest= (graph_shortest-ref_shortest)/(max_avg_distance-ref_shortest)
    #     return graph_shortest
    else:
         return 0

def modularity(graph,normalized=True):
    """Returns the modularity of the graph

    If normalized=True, it is computed as the difference between the modularity of the partition of highest modularity found by Louvain algorithm on this graph and the modularity of a random graph with the same number of nodes and edges, divided by the difference between the modularity of the random graph and the maximum possible modularity (1). If normalized is false, it simply returns the modularity of the graph.
    
    Args:
        graph (_type_): a graph
        normalized (bool, optional): if True, returns a normalized modularity. Defaults to True.

    Returns:
        _type_: The modularity of the graph
    """
    mod=nx.algorithms.community.modularity(graph,nx.algorithms.community.louvain_communities(graph))
    if normalized:
        ref_model = stn.Graph_generator.ER(graph.number_of_nodes(),nx.density(graph))
        ref_mod = nx.algorithms.community.modularity(ref_model,nx.algorithms.community.louvain_communities(ref_model))
        mod = max(0,(mod-ref_mod)/(1-ref_mod))
    return mod
