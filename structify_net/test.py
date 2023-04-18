

def has_giant_component(graph,threshold=0.9):
    """
    Plot a matrix of the graph, ordered by nodePair_order
    graph: a networkx graph
    nodePair_order: a list of node pairs, from the most likely to the less likely
    nodeOrder: a list of nodes, ordered by the order in which they should appear in the matrix
    """
    return giant_component_ratio(graph)>threshold

def giant_component_ratio(graph):
    """
    Plot a matrix of the graph, ordered by nodePair_order
    graph: a networkx graph
    nodePair_order: a list of node pairs, from the most likely to the less likely
    nodeOrder: a list of nodes, ordered by the order in which they should appear in the matrix
    """
    #nb_nodes=graph.number_of_nodes()
    #largest_CC= len(max(nx.connected_components(graph), key=len))
    #ratio_CC=largest_CC/nb_nodes
    return graph