import scipy
import scipy.spatial
import itertools
import random
import math
import networkx as nx
import numpy as np
import structify_net as stn
import perlin_noise 

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
    
    Returns a rank model based on the Erdos-Renyi model. The Erdos-Renyi model is a random graph model where each pair of nodes is connected with probability p. For a rank model, all pairs of nodes have the same probability.

    Args:
        nodes (_type_): describe nodes of the graphs, either as a networkx graph (node names and node attributes are preserved) or an integer (number of nodes)

    Returns:
        :class:`structify_net.Rank_model`:: The corresponding rank model
    """
    return stn.Rank_model(nodes,lambda x,y,z: 0)




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


def sort_distances(nodes,dimensions=1,distance="euclidean"):
    """Rank model based on the distance between nodes
    
    Also called spatial or geometric network, this rank model is based on the distance between nodes. The distance is defined by the distance function, and the dimensions are defined by the attributes of the nodes. By default, a single dimension is used, and the distance is the euclidean distance.

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

    positions={n:[g.nodes[n][d] for d in dimensions] for n in g.nodes}
    def rank_function(u,v,_):
        return distance(positions[u],positions[v])
    
    node_order = lambda g:sorted(g.nodes,key=lambda n: g.nodes[n][dimensions[0]])
    return stn.Rank_model(g,rank_function,node_order_function=lambda g: node_order(g))

def sort_perlin_noise(nodes=10,octaves=None):
    """Rank model based on Perlin noise

    Perline noise is a type of gradiant noise frequently used in computer graphics to create images with a realistic feel, such as textures and landscapes. we use it to generate an adjacency matrix, from the upper-triangle of a 2d image having as many pixels as there are nodes in the graph. The $R'$ rank score is the black intensity of the pixel. In practice, Perlin noise tends to create continuous shapes of lower and higher values. As with SBM, this generator tends to create stronger relations between some groups of nodes with some other groups of nodes, although the groups are fuzzy, and not necessarily assortative. Perlin noise has a parameter, called octaves, allowing to add smaller scale structures on top of each other.


    Args:
        nodes (int, optional): _description_. Defaults to 10.
        octaves (_type_, optional): Octave parameter of the Perlin noise. The higher the value, the finer the structure. Defaults to None, meaning octaves = int(ln(n))

    Returns:
        :class:`structify_net.Rank_model`:: The corresponding rank model
    """
    if not isinstance(nodes,nx.Graph):
        g=  _n_to_graph(nodes)
    else:
        g=nodes
    n=len(g.nodes)
    
    if octaves==None:
        octaves = int(np.log(n))
    noise = perlin_noise.PerlinNoise(octaves=octaves)

    
    def rank_function(u,v,g):
        return noise([u/n,v/n])
    return stn.Rank_model(g,rank_function)


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
    def rank_function(u,v,g):
        return 2 if blocks[u]==blocks[v] else 0
    
    def node_order_function(g):
        return sorted(g.nodes,key=lambda n: g.nodes[n]["block1"])
    
    return stn.Rank_model(g,rank_function,node_order_function=lambda g: node_order_function(g),sort_descendent=True)

def _flatten(l):
    return [item for sublist in l for item in sublist]

def sort_overlap_communities(nodes,blocks=None):
    """Rank model based on overlapping communities

    This rank model is based on overlapping communities. The communities are defined by the blocks argument. It can be either a list of lists, where each list is a community, or an integer corresponding to the number of communities. In the latter case, each node belongs to two communities, and the affiliations are chosen such as each community has half of its nodes shared with another community c1 and the other half shared with another community c2.
    
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

    

    all_blocks={}
    for b in blocks:
        affiliations=nx.get_node_attributes(g,b)
        all_blocks[b]=affiliations

    def rank_function(u,v,g):
        for b in blocks:
            if all_blocks[b][u]==all_blocks[b][v]:
                return 1
                break
        return 0

    node_order = lambda g: sorted(nx.get_node_attributes(g,"block1").items(), key=lambda e: e[1],reverse=False)
    return stn.Rank_model(g,rank_function,node_order_function=lambda g: [e[0] for e in node_order(g)],sort_descendent=True)

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

    c_size=math.ceil((2*m)/n+1) #Because we know the average degree, so the correst size for a clique
    nb_c=math.floor(n/c_size)
    missing=n-c_size*nb_c
    affil = [[c_label]*c_size for c_label in range(nb_c)]
    affil = [item for sublist in affil for item in sublist]+[nb_c-1]*missing
    blocks={i:affil[i] for i in range(n)}
    def rank_function(u,v,g):
        return 1 if blocks[u]==blocks[v] else 0
    return stn.Rank_model(g,rank_function,sort_descendent=True)


def sort_stars(nodes):
    """A rank model based on a star structure
    
     This rank model sorts pairs of nodes such as all pairs of nodes of node n1 are first, then all pairs of nodes of node n2, etc.
     
    Args:
        nodes (_type_): Describe nodes. Can be either a networkx graph (node names and node attributes are preserved) or an integer (number of nodes)

    Returns:
        :class:`structify_net.Rank_model`:: The rank model
    """
    if not isinstance(nodes,nx.Graph):
        g=_n_to_graph(nodes)
    else:
        g=nodes
    n=len(g.nodes)
    def R(u,v,g):
        return u*n+v
    return stn.Rank_model(g,R,sort_descendent=False)
# def sort_stars(nodes):
   
#     if not isinstance(nodes,nx.Graph):
#         g=_n_to_graph(nodes)
#     else:
#         g=nodes

#     sorted_pairs=itertools.combinations(g.nodes,2)
#     return stn.Rank_model(list(sorted_pairs), g)


































def sort_core_distance(nodes,dimensions=1,distance="euclidean"):
    """Rank model based on the sum of the distance of their nodes to the center of the space.
    
    Core periphery structure is another well-known type of organization for complex systems. This organization is often modeled using blocks, one block being the dense core, another block, internally sparse, represent the periphery, and the density between the two blocks is set at an intermediate value. To illustrate the flexibility of the Rank approach, we propose a soft-core alternative, the coreness dissolving progressively into a periphery. To do so, we consider nodes embedded into a space, as for the spatial structure --random 1d positions by default. The node-pair rank score is computed as the inverse of the product of 3 distances: the distances from both nodes to the center, and the distance between the two nodes. As a consequence, when two nodes belong to the center, they are very likely to be connected; two nodes far from the center are unlikely to be connected, unless if they are extremely close from each other. 
    \[
    R'(u,v)=d(W_u,W_v)d(W_u,\mathbf{0})d(W_v,\mathbf{0})
    \]
    With $\mathbf{0}$ the vector corresponding to the center of the location considered as the core of the space.
    
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

    positions={n:[g.nodes[n][d] for d in dimensions] for n in g.nodes}
    def core_distance(u,v,_):
        return distance(positions[u],[0.5]*len(dimensions))*distance(positions[v],[0.5]*len(dimensions))*distance(positions[u],positions[v])
    
    def node_order_function(g):
        return sorted(g.nodes,key=lambda n: g.nodes[n]["d1"],reverse=False)

    return stn.Rank_model(g,core_distance,sort_descendent=False,node_order_function=node_order_function)


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
    n=len(g.nodes)
    def my_dist(u,v,g):
        if ((v-u)%(n-(k/2))<=k/2):
            return 0
        return 2

    return stn.Rank_model(g,my_dist,sort_descendent=False)

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

    all_distances = {x:v for x,v in nx.all_pairs_shortest_path_length(binary_tree)}

    def distance(u,v,_):
        return all_distances["temp_"+str(u)]["temp_"+str(v)]
    return stn.Rank_model(g,distance,sort_descendent=False)

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
    height=_determine_tree_height(nb_nodes,d)
    binary_tree = nx.balanced_tree(d,height)
    leaves_ids=list(binary_tree.nodes)[:nb_nodes]
    binary_tree = nx.relabel_nodes(binary_tree,{n:"temp_"+str(i) for i,n in enumerate(leaves_ids)})
    all_distances = {x:v for x,v in nx.all_pairs_shortest_path_length(binary_tree)}
    def distance(u,v,_):
        return all_distances["temp_"+str(u)]["temp_"+str(v)]
    return stn.Rank_model(g,distance,sort_descendent=False)

def _determine_tree_height(n,d):
    tree_size=1
    height=0
    while tree_size<n:
        tree_size+=d**(height+1)
        height+=1
    
    return height

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

    def R_nestedness(u,v,_):
        return u+v
    return stn.Rank_model(g,R_nestedness,sort_descendent=False)

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
    height=_determine_tree_height(nb_nodes,d)
    binary_tree = nx.balanced_tree(d,height)
    leaves_ids=list(binary_tree.nodes)[:nb_nodes]
    binary_tree = nx.relabel_nodes(binary_tree,{n:"temp_"+str(i) for i,n in enumerate(leaves_ids)})

    pairs=itertools.combinations(g.nodes,2)
    all_distances = {x:v for x,v in nx.all_pairs_shortest_path_length(binary_tree) }
    depths = nx.shortest_path_length(binary_tree,"temp_"+str(0))
    
    def height_from_bottom(u):
        return height-depths["temp_"+str(u)]
    
    def child_of(u,v):
        return all_distances["temp_"+str(u)]["temp_"+str(v)]==abs(depths["temp_"+str(u)]-depths["temp_"+str(v)])
    
    def child_score(u,v):

        return min(height_from_bottom(u),height_from_bottom(v))+min(depths["temp_"+str(u)],depths["temp_"+str(v)])
    
    def family_score(u,v):
        if depths["temp_"+str(u)]==depths["temp_"+str(v)]: #same height
            return (all_distances["temp_"+str(u)]["temp_"+str(v)]-2)+height_from_bottom(u)
        else:
            return height+all_distances["temp_"+str(u)]["temp_"+str(v)]
    def R_fractal(u,v,_):
        return child_score(u,v) if child_of(u,v) else family_score(u,v)
    return stn.Rank_model(g,R_fractal,sort_descendent=False)                                            


























def sort_fractal_star(nodes,d=2):
    """A rank model based on a fractal structure
    
    This structure is designed to maximize the star structure of the network. The network is embedded in a binary tree. The order of the pairs is based on the distance between the hierarchical levels.

    Args:
        nodes (_type_): Describe nodes. Can be either a networkx graph (node names and node attributes are preserved) or an integer (number of nodes)
        d (int, optional): degree of the binary tree. Defaults to 2.

    Returns:
        :class:`structify_net.Rank_model`:: rank model
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

    heights = nx.shortest_path_length(binary_tree,"temp_"+str(0))
    def rank(u,v,_):
        return abs(heights["temp_"+str(u)]-heights["temp_"+str(v)])
    return stn.Rank_model(g,rank,sort_descendent=True)


























all_models_no_param={"ER":sort_ER,"spatial":sort_distances,"spatialWS":sort_spatial_WS,"blocks_assortative":sort_blocks_assortative,"overlapping_communities":sort_overlap_communities,
             "nestedness":sort_nestedness,"maximal_stars":sort_stars,"core_distance":sort_core_distance,"fractal_leaves":sort_fractal_leaves,
             "fractal_root":sort_fractal_root,"fractal_hierarchy":sort_fractal_hierarchical,"fractal_star":sort_fractal_star,"perlin_noise":sort_perlin_noise}

all_models_with_m={"disconnected_cliques":sort_largest_disconnected_cliques}
#all_models={ **all_models_no_param, **all_models_with_m }#|all_models_with_m

def get_all_rank_models(n,m):
    """Returns a dictionary of all rank models

    Args:
        n (_type_): number of nodes
        m (_type_): number of edges. Only used for models with a parameter m

    Returns:
        Dictionary of rank models {name:rank_model}
    """
    to_return = {name:f(n) for name,f in all_models_no_param.items()}
    for name,f in all_models_with_m.items():
        to_return[name]=f(n,m)
    #to_return = to_return|{name:f(n,m) for name,f in all_models_with_m.items()}
    return to_return

#def get_all_generators(n,m):
#    to_return = {name:f(n).get_generator for name,f in all_models_no_param.items()}
#    return to_return