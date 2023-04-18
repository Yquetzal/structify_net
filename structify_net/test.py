import numpy as np
import structify_net as stn
import networkx as nx


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