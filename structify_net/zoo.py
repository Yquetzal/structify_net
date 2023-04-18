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












# def ER_generator(n,m=None,p=None):
#     """Return a graph generator based on the Erdos-Renyi model

#     Returns a graph generator based on the Erdos-Renyi model. The Erdos-Renyi model is a random graph model where each pair of nodes is connected with equal probability p. 
    
#     Either the number of edges m or the probability p must be specified. If both are specified, m takes precedence.
    
#     Args:
#         n (_type_): number of nodes
#         m (_type_, optional): number of edges. Defaults to None.
#         p (_type_, optional): probability of an edge. Defaults to None.

#     Raises:
#         ValueError: _description_
        
#     Returns:
#         :class:`structify_net.Graph_generator`:: The corresponding graph generator
#     """
#     if m==None and p==None:
#         raise ValueError("Either m or p must be specified")
#     if m==None:
#         m=int(n*p)
#     stn.Graph_generator.ER(n,m)


def _assign_nominal_attributes(blocks,nb_nodes=None,g=None,name="block1"):
    if g==None:
        g=nx.Graph()
        g.add_nodes_from(range(nb_nodes))
    if nb_nodes==None:
        nb_nodes=len(g.nodes)
        
        
    if isinstance(blocks,int):
        node_per_block=int(nb_nodes/blocks)
        affils=[]
        for b in range(blocks):
            affils+=[b]*node_per_block
        unaffiled=nb_nodes-len(affils)
        if unaffiled>0:
            affils+=list(range(blocks))[:unaffiled]
    
    random.shuffle(affils)
    n2affil={n:affils[i] for i,n in enumerate(g.nodes)}
    nx.set_node_attributes(g,n2affil,name)
    return(g)



#Spatial/Geometric network, homophily (Ordinal)
#Nodes are ranked according to the euclidean distance between their attributes. 
#Typically, 1Dimension for homophily, 2 for spatial.
def sort_distances(nodes,dimensions=1,distance="euclidean"):
    """Rank model based on the distance between nodes
    
    Also called spatial or geometric network, this rank model is based on the distance between nodes. The distance is defined by the distance function, and the dimensions are defined by the attributes of the nodes.

    Args:
        nodes (_type_): describe nodes of the graphs, either as a networkx graph (node names and node attributes are preserved) or an integer (number of nodes)
        dimensions (int, optional): number of dimensions in which to embed the graph. Defaults to 1.
        distance (_type_, optional): distance function. Defaults to euclidean.

    Returns:
        :class:`structify_net.Rank_model`:: The corresponding rank model
    """
    
    if distance=="euclidean":
        distance = scipy.spatial.distance.euclidean
    if not isinstance(nodes,nx.Graph):
        g=_n_to_graph(nodes)
    else:
        g=nodes

    if isinstance(dimensions,int):
        g = _assign_ordinal_attributes(len(g.nodes),dimensions)
        dimensions=["d"+str(i_dim+1) for i_dim in range(dimensions)]
    sorted_pairs=itertools.combinations(g.nodes,2)
    positions={n:[g.nodes[n][d] for d in dimensions] for n in g.nodes}
    sorted_pairs={(u,v):distance(positions[u],positions[v]) for u,v in sorted_pairs}
    sorted_pairs = sorted(sorted_pairs.items(), key=lambda e: e[1],reverse=False)

    sorted_pairs=[e[0] for e in sorted_pairs]
    node_order = sorted(nx.get_node_attributes(g,"d1").items(), key=lambda e: e[1],reverse=False)
    return stn.Rank_model(sorted_pairs, g,node_order=[e[0] for e in node_order])

#Defining assortative blocks/communities. 
#Be carefull that pairs are ordered both inside and outside blocks (favor low number nodes)
def sort_blocks_assortative(nodes,blocks=None):
    """Rank model based on assortative blocks
    
    This rank model is based on assortative blocks. The blocks are defined by the blocks argument. It can be either a list of lists, where each list is a block, or an integer, in which case the nodes are randomly assigned to blocks.
    
    Edge pairs inside blocks are ordered before edge pairs outside blocks.

    Args:
        nodes (_type_): Describe nodes of the graphs, either as a networkx graph (node names and node attributes are preserved) or an integer (number of nodes)
        blocks (_type_, optional): Blocks definition. Can be either a list of lists, where each list is a block, or an integer, in which case the nodes are randomly assigned to the corresponding number of equal size blocks. Defaults to None.

    Returns:
        :class:`structify_net.Rank_model`: The corresponding rank model
    """
    if not isinstance(nodes,nx.Graph):
        g=_n_to_graph(nodes)
    else:
        g=nodes

    sorted_pairs=itertools.combinations(g.nodes,2)
    if blocks==None:
        blocks=math.ceil(math.sqrt(len(g.nodes)))
    if isinstance(blocks,int):
        g = _assign_nominal_attributes(blocks,len(g.nodes),g)
        blocks="block1"
    if isinstance(blocks,list):
        affiliations= {n:i  for i,b in enumerate(blocks) for n in b}
        nx.set_node_attributes(g,affiliations,"block1")
        blocks="block1"
    
    blocks=nx.get_node_attributes(g,blocks)
    sorted_pairs={(u,v): 2+random.random() if blocks[u]==blocks[v] else 0+random.random() for u,v in sorted_pairs}
    sorted_pairs = sorted(sorted_pairs.items(), key=lambda e: e[1],reverse=True)
    sorted_pairs=[e[0] for e in sorted_pairs]

    node_order = sorted(nx.get_node_attributes(g,"block1").items(), key=lambda e: e[1],reverse=False)
    return stn.Rank_model(sorted_pairs, g,node_order=[e[0] for e in node_order])

def _flatten(l):
    return [item for sublist in l for item in sublist]

def sort_overlap_communities(nodes,blocks=None):
    """Rank model based on overlapping communities

    This rank model is based on overlapping communities. The communities are defined by the blocks argument. It can be either a list of lists, where each list is a community, or an integer corresponding to the number of communities. In the latter case, each node corresponds to two communities, and the affiliations are chosen such as each community has half of its nodes shared with another community c1 and the other half shared with another community c2.
    
    Args:
        nodes (_type_): Describe nodes of the graphs, either as a networkx graph (node names and node attributes are preserved) or an integer (number of nodes)
        blocks (_type_, optional): Describe communities. Can be either a list of lists, where each list is a community, or an integer. Defaults to None.

    Returns:
        :class:`structify_net.Rank_model`:: A rank model
    """
    if not isinstance(nodes,nx.Graph):
        g=_n_to_graph(nodes)
    else:
        g=nodes
    n=len(g.nodes)
    if blocks==None:
        blocks=math.ceil(math.sqrt(n))
        
    if isinstance(blocks,int):
        block_size=math.ceil(n/(blocks/2))
        nb_blocks_by_level=int(blocks/2)
        block1=_flatten([[i]*block_size for i in range(nb_blocks_by_level)])[:n]
        largest_id=max(block1)
        block2=_flatten([[i+largest_id+1]*block_size for i in range(nb_blocks_by_level)])[:n]
        block2 = block2[int(block_size/2):]+block2[:int(block_size/2)]
        nx.set_node_attributes(g,{i:b for i,b in enumerate(block1)},"block1")
        nx.set_node_attributes(g,{i:b for i,b in enumerate(block2)},"block2")
        blocks=["block1","block2"]
        #g = _assign_nominal_attributes(block_id,len(g.nodes),g)
        #block_id="block"
    
    sorted_pairs=itertools.combinations(g.nodes,2)

    all_blocks={}
    for b in blocks:
        affiliations=nx.get_node_attributes(g,b)
        all_blocks[b]=affiliations
    similarity={}
    for u,v in sorted_pairs:
        similarity[(u,v)]=np.random.random()/100
        for b in blocks:
            if all_blocks[b][u]==all_blocks[b][v]:
                similarity[(u,v)]=1+np.random.random()/100
                #similarity[(u,v)]+=1
    sorted_pairs = sorted(similarity.items(), key=lambda e: e[1],reverse=True)
    #sorted_pairs={(u,v): 2+random.random() if blocks[u]==blocks[v] else 0+random.random() for u,v in sorted_pairs}
    sorted_pairs=[e[0] for e in sorted_pairs]

    node_order = sorted(nx.get_node_attributes(g,"block1").items(), key=lambda e: e[1],reverse=False)
    return stn.Rank_model(sorted_pairs, g,node_order=[e[0] for e in node_order])

#Defining assortative blocks/communities. 
#Be carefull that pairs are ordered both inside and outside blocks (favor low number nodes)
def sort_largest_disconnected_cliques(nodes,m):
    """A rank model based on the largest disconnected cliques

    Computes the largest possible number of cliques of size k such that the number of edges is less than m. Then, the rank model is the same as for assortative blocks.
    
    Args:
        nodes (_type_): Nodes of the graph, either as a networkx graph (node names and node attributes are preserved) or an integer (number of nodes)
        m (_type_): number of edges. This is required to compute the largest possible number of cliques.

    Returns:
        :class:`structify_net.Rank_model`:: _description_
    """
    if not isinstance(nodes,nx.Graph):
        g=_n_to_graph(nodes)
    else:
        g=nodes

    sorted_pairs=itertools.combinations(g.nodes,2)
    n=len(g.nodes())
    #(c*(c-1)/2)*k<=m, c*k=n, I want to find the largest possible k
    #nb_c= = (n/m) / 2 + sqrt((n/m)^2/4 - 2)
    c_size=math.ceil((2*m)/n+1) #Because we know the average degree, so the correst size for a clique
    nb_c=math.floor(n/c_size)
    missing=n-c_size*nb_c
    affil = [[c_label]*c_size for c_label in range(nb_c)]
    affil = [item for sublist in affil for item in sublist]+[nb_c-1]*missing
    blocks={i:affil[i] for i in range(n)}
    sorted_pairs={(u,v): 2+random.random() if blocks[u]==blocks[v] else 0+random.random() for u,v in sorted_pairs}
    sorted_pairs = sorted(sorted_pairs.items(), key=lambda e: e[1],reverse=True)
    return stn.Rank_model([e[0] for e in sorted_pairs], g)

#Defining a degree heterogeneous network. Pairs of nodes are sorted such as all pairs of nodes of node n1 are first,
# then all pairs of nodes of node n2, etc. The strongest structure corresponds to a star structure: a few nodes
# are connected to all others, that have only this link.
def sort_stars(nodes):
    """A rank model based on a star structure
    
    A star structure is a few nodes connected to all others, that have only this link.
    

    Args:
        nodes (_type_): Describe nodes. Can be either a networkx graph (node names and node attributes are preserved) or an integer (number of nodes)

    Returns:
        :class:`structify_net.Rank_model`:: The rank model
    """
    if not isinstance(nodes,nx.Graph):
        g=_n_to_graph(nodes)
    else:
        g=nodes

    sorted_pairs=itertools.combinations(g.nodes,2)
    return stn.Rank_model(list(sorted_pairs), g)



































# A proposition of a (continuous) core-periphery organization. Pairs of nodes are sorted according to the sum 
# of the distance of their nodes to the center of the space.
def sort_core_distance(nodes,dimensions=1,distance="euclidean"):
    """Rank model based on the sum of the distance of their nodes to the center of the space.
    
    This is a proposition of a (continuous) core-periphery organization. Pairs of nodes are sorted according to the sum of the distance of their nodes to the center of the space.

    Args:
        nodes (_type_): Describe nodes. Can be either a networkx graph (node names and node attributes are preserved) or an integer (number of nodes)
        dimensions (int, optional): Number of dimensions. Defaults to 1.
        distance (_type_, optional): Distance function. Defaults to euclidean.

    Returns:
        :class:`structify_net.Rank_model`:: The rank model
    """
    
    if distance=="euclidean":
        distance=scipy.spatial.distance.euclidean
    
    if not isinstance(nodes,nx.Graph):
        g=_n_to_graph(nodes)
    else:
        g=nodes
    if isinstance(dimensions,int):
        g = _assign_ordinal_attributes(len(g.nodes),dimensions)
        dimensions=["d"+str(i_dim+1) for i_dim in range(dimensions)]

    sorted_pairs=itertools.combinations(g.nodes,2)
    positions={n:[g.nodes[n][d] for d in dimensions] for n in g.nodes}
    def core_distance(u,v):
        return distance(positions[u],[0.5]*len(dimensions))*distance(positions[v],[0.5]*len(dimensions))
    sorted_pairs={(u,v):core_distance(u,v) for u,v in sorted_pairs}
    sorted_pairs = sorted(sorted_pairs.items(), key=lambda e: e[1],reverse=False)
    sorted_pairs=[e[0] for e in sorted_pairs]
    node_order = sorted(nx.get_node_attributes(g,"d1").items(), key=lambda e: e[1],reverse=False)
    return stn.Rank_model(sorted_pairs, g,[e[0] for e in node_order])

def sort_spatial_WS(nodes,k=10):
    """Rank model based on a spatial Watts-Strogatz model
    
    This rank model reproduce the original Watts-Strogatz model. Each node is connected to its k nearest neighbors in a ring topology.

    Args:
        nodes (_type_): A networkx graph (node names and node attributes are preserved) or an integer (number of nodes)
        k (int, optional): Number of nearest neighbors. Defaults to 10.

    Returns:
        :class:`structify_net.Rank_model`:: The rank model
    """
    if not isinstance(nodes,nx.Graph):
        g=_n_to_graph(nodes)
    else:
        g=nodes

    sorted_pairs=itertools.combinations(g.nodes,2)
    def my_dist(u,v):
        if ((v-u)%(len(g.nodes)-(k/2))<=k/2):
            return 0+random.random()
        return 2+random.random()
    sorted_pairs={(u,v):my_dist(u,v) for u,v in sorted_pairs}
    sorted_pairs = sorted(sorted_pairs.items(), key=lambda e: e[1],reverse=False)
    return stn.Rank_model([e[0] for e in sorted_pairs], g)

def sort_fractal_leaves(nodes,d=2):
    """A rank model based on a fractal structure
    
    The order of the pairs is based on the distance between the leaves of a binary tree. The number of leaves is the number of nodes in the network. The distance between two leaves is the number of edges between them in the binary tree.

    Args:
        nodes (_type_): Describe nodes. Can be either a networkx graph (node names and node attributes are preserved) or an integer (number of nodes)
        d (int, optional): Degree of the binary tree. Defaults to 2.

    Returns:
        :class:`structify_net.Rank_model`:: A rank model
    """
    if not isinstance(nodes,nx.Graph):
        g=_n_to_graph(nodes)
    else:
        g=nodes

    nb_nodes=len(g.nodes)
    height = math.ceil(math.log(nb_nodes, d))
    binary_tree = nx.balanced_tree(d,height)
    leaves_ids=list(binary_tree.nodes)[-nb_nodes:]
    binary_tree = nx.relabel_nodes(binary_tree,{n:"temp_"+str(i) for i,n in enumerate(leaves_ids)})

    pairs=itertools.combinations(g.nodes,2)
    all_distances = {x:v for x,v in nx.all_pairs_shortest_path_length(binary_tree)}
    #print(all_distances)
    sorted_pairs = {(u,v):all_distances["temp_"+str(u)]["temp_"+str(v)]+random.random()/10 for (u,v) in pairs}
    sorted_pairs = sorted(sorted_pairs.items(), key=lambda e: e[1],reverse=False)
    return stn.Rank_model([e[0] for e in sorted_pairs], g)

def sort_fractal_root(nodes,d=2):
    """A rank model based on a fractal structure
    
    The network is embedded in a binary tree. The order of the pairs is based on the distance between the nodes in the tree. Nodes are assigned starting from the root of the tree.

    Args:
        nodes (_type_): Describe nodes. Can be either a networkx graph (node names and node attributes are preserved) or an integer (number of nodes)
        d (int, optional): degree of the binary tree. Defaults to 2.

    Returns:
        :class:`structify_net.Rank_model`:: The rank model
    """
    if not isinstance(nodes,nx.Graph):
        g=_n_to_graph(nodes)
    else:
        g=nodes

    nb_nodes=len(g.nodes)
    height = math.ceil(math.log(nb_nodes, d)+1)
    binary_tree = nx.balanced_tree(d,height-1)
    leaves_ids=list(binary_tree.nodes)[:nb_nodes]
    binary_tree = nx.relabel_nodes(binary_tree,{n:"temp_"+str(i) for i,n in enumerate(leaves_ids)})

    pairs=itertools.combinations(g.nodes,2)
    all_distances = {x:v for x,v in nx.all_pairs_shortest_path_length(binary_tree)}
    sorted_pairs = {(u,v):all_distances["temp_"+str(u)]["temp_"+str(v)]+random.random()/10 for (u,v) in pairs}
    sorted_pairs = sorted(sorted_pairs.items(), key=lambda e: e[1],reverse=False)
    return stn.Rank_model([e[0] for e in sorted_pairs], g)

def sort_nestedness(nodes):
    """Rank model based on nestedness
    
    A nestedness network is a particular type of network where the nodes are organized in a hierarchy. Top nodes are connected to bottom nodes. Bottom nodes are connected to other bottom nodes. The order of the pairs is based on the distance between the nodes in the hierarchy.

    Args:
        nodes (_type_): Describe nodes. Can be either a networkx graph (node names and node attributes are preserved) or an integer (number of nodes)

    Returns:
        :class:`structify_net.Rank_model`:: The rank model
    """
    if not isinstance(nodes,nx.Graph):
        g=_n_to_graph(nodes)
    else:
        g=nodes

    sorted_pairs=itertools.combinations(g.nodes,2)
    sorted_pairs = sorted(list(sorted_pairs),key=lambda x: x[0]+x[1])
    return stn.Rank_model(sorted_pairs, g)

def sort_fractal_hierarchical(nodes,d=3):
    """Rank model based on a fractal structure
    
    This structure is designed to maximize the hierarchical structure of the network. The network is embedded in a binary tree. The order of the pairs is based on two factors: the distance between nodes in the tree at a same hierarchical level and the distance between the hierarchical levels.

    Args:
        nodes (_type_): Describe nodes. Can be either a networkx graph (node names and node attributes are preserved) or an integer (number of nodes)
        d (int, optional): degree of the binary tree. Defaults to 3. 3 allows to have many triangles.

    Returns:
        :class:`structify_net.Rank_model`:: The rank model
    """
    if not isinstance(nodes,nx.Graph):
        g=_n_to_graph(nodes)
    else:
        g=nodes

    nb_nodes=len(g.nodes)
    height = math.ceil(math.log(nb_nodes, d)+1)
    binary_tree = nx.balanced_tree(d,height-1)
    leaves_ids=list(binary_tree.nodes)[:nb_nodes]
    binary_tree = nx.relabel_nodes(binary_tree,{n:"temp_"+str(i) for i,n in enumerate(leaves_ids)})

    pairs=itertools.combinations(g.nodes,2)
    all_distances = {x:v for x,v in nx.all_pairs_shortest_path_length(binary_tree) }
    #degrees = {k:v for k,v in binary_tree.degree}
    heights = nx.shortest_path_length(binary_tree,"temp_"+str(0))
    
    def child_of(u,v):
        return all_distances["temp_"+str(u)]["temp_"+str(v)]==abs(heights["temp_"+str(u)]-heights["temp_"+str(v)])
    
    def child_score(u,v):
        #return height-max(heights["temp_"+str(u)],heights["temp_"+str(v)])
        return height-1-abs(heights["temp_"+str(u)]-heights["temp_"+str(v)])
    
    def family_score(u,v):
        return (all_distances["temp_"+str(u)]["temp_"+str(v)]-2)*1000+height-1-max(heights["temp_"+str(u)],heights["temp_"+str(v)])#+child_score(u,v)
                                                                  
    sorted_pairs = {(u,v):child_score(u,v) if child_of(u,v) else family_score(u,v) for (u,v) in pairs}
    sorted_pairs = sorted(sorted_pairs.items(), key=lambda e: e[1],reverse=False)
    return stn.Rank_model([e[0] for e in sorted_pairs], g)

