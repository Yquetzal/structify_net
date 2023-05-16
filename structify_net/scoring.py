import math
import networkx as nx
import numpy as np
import scipy
#from scipy.stats import spearmanr
import pandas as pd
import structify_net as stn
#from structify_net.structureClasses import Rank_model, Graph_generator
import numbers
from tqdm.auto import tqdm
#import tqdm

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

def _gini_coefficient(x):
    x=np.array(x)
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))

def degree_heterogeneity(graph):
    """Compute the degree heterogeneity of the graph
    
    Args:
        graph (nx.Graph): A graph
    Returns:
        float: The degree heterogeneity of the graph
    """
    degrees = [d for n, d in graph.degree()]
    heterogeneity = _gini_coefficient(degrees)
    return heterogeneity
    #avg_degree=statistics.mean([d for n, d in graph.degree()]) 
    #degree_heterogeneity=statistics.stdev(degrees)-statistics.sqrt(avg_degree)

def is_degree_heterogeneous(graph):
    """Returns True if the graph is degree heterogeneous, False otherwise
    
    A graph is considered degree heterogeneous if its degree heterogeneity is greater than the average between the degree heterogeneity of a random graph with the same number of nodes and edges, and a power law graph with the same number of nodes and edges.
    """
    avg_degree = np.average([d for n, d in graph.degree()])
    reference_threshold = _gini_coefficient(np.random.poisson(avg_degree,len(graph.nodes)))
    power_law_threshold = _gini_coefficient(nx.utils.powerlaw_sequence(len(graph.nodes),3))
    #print("-",degree_heterogeneity(graph),reference_threshold,power_law_threshold)
    return degree_heterogeneity(graph)>((reference_threshold+power_law_threshold)/3)

def _robustness_func(g:nx.Graph):
    nb_nodes=len(g.nodes)
    hub_sorted = [k for k,v in sorted(g.degree, key=lambda x: x[1], reverse=True)]
    #print(hub_sorted)
    for i in [1,2,3,4,5,6,7,8,9,10,20,30,40,49]:
        #print()
        g2 =g.copy()
        g2.remove_nodes_from(hub_sorted[:int(i/100*nb_nodes)])
        largest_CC= len(max(nx.connected_components(g2), key=len))
        if largest_CC<nb_nodes*0.5:
            return i/50
    return 50/50

def robustness(graph):
    """Robustness of the graph
    
    Robustness is defined as the percentage of nodes that need to be removed to disconnect the graph. It is computed by removing nodes from the graph, starting from the most connected nodes, until the graph is disconnected. The percentage of nodes removed is then returned.

    Args:
        graph (_type_): a graph

    Returns:
        _type_: robustness of the graph
    """
    return _robustness_func(graph)

def degree_assortativity(graph,normalized=False,positive_only=True,disassortativity=False):
    """Degree assortativity coefficient of the graph
    
    The degree assortativity coefficient is computed as the Pearson correlation coefficient between the degrees of the nodes. If normalized=True, it is normalized to be between 0 and 1. If positive_only=True, it is set to 0 if negative. If disassortativity=True, it is set to -1 if positive and 1 if negative.

    Args:
        graph (_type_): _description_
        normalized (bool, optional): _description_. Defaults to False.
        positive_only (bool, optional): _description_. Defaults to True.
        disassortativity (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if not is_degree_heterogeneous(graph):
        deg_coeff=0
    else:
        degrees = [d for n, d in graph.degree()]
        deg_coeff=0
        for k in degrees:
            if k!=degrees[0]:  
                deg_coeff=nx.degree_assortativity_coefficient(graph)
                break
    if normalized:
        deg_coeff=deg_coeff/2+0.5
    if disassortativity:
        deg_coeff=-deg_coeff
    if positive_only:
        deg_coeff=max(0,deg_coeff)
    return deg_coeff

def hierarchy(graph,normalized=False,positive_only=True):
    """Hierarchy of the graph
    
    The hierarchy is computed as the Spearman correlation coefficient between the degrees and the clustering coefficients of the nodes. If normalized=True, it is normalized to be between 0 and 1. If positive_only=True, it is set to 0 if negative.

    Args:
        graph (_type_): A graph
        normalized (bool, optional): If True, the hierarchy is normalized to be between 0 and 1. Defaults to False.
        positive_only (bool, optional): IF True, the hierarchy is set to 0 if negative. Defaults to True.

    Returns:
        _type_: hierarchy of the graph
    """
    degrees = [d for n, d in graph.degree()]
    clusterings = [d for n, d in nx.clustering(graph).items()]
    if not is_degree_heterogeneous(graph):
        hierarchical=0
    else:
        hierarchical=0
        constant_degrees=True
        constant_clustering=True
        for i in range(len(degrees)):
            if degrees[i]!=degrees[0]:
                constant_degrees=False
            if clusterings[i]!=clusterings[0]:
                constant_clustering=False
            if not constant_degrees and not constant_clustering:
                hierarchical=-scipy.stats.spearmanr(degrees,clusterings).correlation
                break
            
        if np.isnan(hierarchical):
            hierarchical=0
    if normalized:
        hierarchical=hierarchical/2+0.5
    if positive_only:
        hierarchical=max(0,hierarchical)
    return hierarchical

def boundaries(graph,normalized=True):
    """Boundaries of the graph

    The boundaries are computed as the average of the density of the external edges of each community, and the density of the internal edges of each community. If normalized=True, it is normalized to be between 0 and 1.
    
    Args:
        graph (_type_): A graph
        normalized (bool, optional): If True, the boundaries are normalized to be between 0 and 1. Defaults to True.

    Returns:
        _type_: _description_
    """
    coms=nx.algorithms.community.louvain_communities(graph)
    average_boundary=0
    for c in coms:
        volume=nx.volume(graph,c)
        if volume==0:
            continue
        external_edges=nx.cut_size(graph,c)
        internal_edges=(volume-external_edges)/2

        density_extern=external_edges/((graph.number_of_nodes()-len(c))*len(c))
        density_intern=2*internal_edges/((len(c)-1)*len(c))
        graph_density=nx.density(graph)

        #print("--",volume,external_edges,internal_edges,graph.number_of_nodes()-len(c),len(c))
        #print("---",density_extern,density_intern,graph_density)
        average_boundary_log=((density_intern-graph_density)-(density_extern-graph_density))#/((1-graph_density)-(0-graph_density))
        average_boundary+=average_boundary_log*len(c)
        #print("----",average_boundary)

        #intern_gain=internal_edges-(graph_density*len(c)*(len(c)-1)/2)
        #extern_gain=(graph_density*len(c)*(graph.number_of_nodes()-len(c)))-external_edges
        #gain=intern_gain+extern_gain
        
        #fraction_intern = 1-external_edges/volume
        #expected_external_edges = nx.density(graph)*len(c)*(graph.number_of_nodes()-len(c))
        #expected_volume=nx.density(graph)*len(c)*(len(c)-1)/2
        #expected_fraction_intern = 1-(expected_external_edges/expected_volume)
        #corrected_fraction_intern = (external_edges-expected_external_edges)/(volume-expected_volume)
        #corrected_fraction_intern = (fraction_intern-expected_fraction_intern)/(1-expected_fraction_intern)

        #average_boundary+=(density_intern-graph_density)*len(c)
    average_boundary=average_boundary/(graph.number_of_nodes())

    return average_boundary

def coreness(graph,normalized=True):
    """Coreness of the graph

    Args:
        graph (_type_): A graph
        normalized (bool, optional):IF True, the coreness is normalized to be between 0 and 1. Defaults to True.

    Returns:
        _type_: _description_
    """
    coreness = max(list(nx.core_number(graph).values()))
    
    #max_possible_coreness = math.floor(1/2 * (math.sqrt( 8*graph.number_of_edges())+1))
    if normalized:
        max_possible_coreness = math.floor(math.sqrt(graph.number_of_edges()))
        coreness= coreness/max_possible_coreness
    return coreness

def compute_all_scores(graph):
    """Compute all scores for a graph
    
    Args:
        graph (_type_): A graph
    
    Returns:
        :class:`pd.DataFrame`: A dictionary with the scores
        
    """
    to_return = {}
    for name,func in default_scores.items():
        to_return[name]=func(graph)
    return to_return

def scores_for_graphs(graphs,scores=None,latex_names=True):
    """Compute scores for a list of graphs

    Args:
        graphs (_type_): A dictionary of graphs such as {name:graph}
        scores (_type_, optional): A dictionary of scores such as {name:score}. Defaults to None.
        latex_names (bool, optional): If True, the names of the scores are latex formulas. Defaults to True.
    """
    records=[]
    if scores is None:
        scores = get_default_scores()
    for graph_name,graph in graphs.items():
        line=[]
        for score_name,func in scores.items():
            line.append(func(graph))
        records.append([graph_name]+line)
    
    df = pd.DataFrame(records,columns=["name"]+list(scores.keys()))
    if latex_names:
        cols=df.columns
        cols=_names2latex_list(cols)
        df.columns=cols
    return(df)




def scores_for_generators(generators,scores=None,runs=1,details=False,latex_names=True):
    """Scores for a list of generators

    Args:
        generators (_type_): A dictionary of generators such as {name:generator}
        scores (_type_, optional): a dictionary of scores such as {name:score}. Defaults to None.
        runs (int, optional): Number of runs. Defaults to 1.
        details (bool, optional): If True, the results of each run are returned. Defaults to False.
        latex_names (bool, optional): If True, the names of the scores are latex formulas. Defaults to True.

    Returns:
        _type_: _description_
    """
    to_return = pd.DataFrame()
    for i in tqdm(range(runs),desc="Run",leave=False, position=0,):
        graphs = {name:generator.generate() for name,generator in generators.items()}
        results= scores_for_graphs(graphs,scores=scores,latex_names=latex_names)
        to_return = pd.concat([to_return,results])
        #scores["run"]=[i]*len(scores)
    if details:
        return to_return
    to_return=to_return.groupby("name").mean().reset_index()
    return to_return


def scores_for_rank_models(rank_models,m,scores=None,epsilons=0,runs=1,details=False,latex_names=True):
    """Scores for a list of rank models

    Args:
        rank_models (_type_): A dictionary of rank models such as {name:rank_model}
        m (_type_): Number of edges
        scores (_type_, optional): A dictionary of scores such as {name:score}. Defaults to None.
        epsilons (int, optional): A list of epsilons. Defaults to 0.
        runs (int, optional): number of runs. Defaults to 1.
        details (bool, optional): If True, the results of each run are returned. Defaults to False.
        latex_names (bool, optional): IF True, the names of the scores are latex formulas. Defaults to True.

    Returns:
        _type_: dataframe with the scores
    """
    all_generators={}
    if not isinstance(rank_models,dict):
        rank_models = {"model":rank_models}
    if isinstance(epsilons,numbers.Number):
        if epsilons<=1:
            epsilons=[epsilons]
        else:
            epsilons=[0]+list(np.logspace(-4,0,epsilons-1))
    all_dfs=[]
    
    pbar = tqdm(epsilons, desc="Epsilon: ",position=0,leave=False)
    for eps in pbar:
    #for eps in epsilons:
        pbar.set_description(f"Epsilon: {round(eps,4)}")
        for name,rank_model in rank_models.items():
            all_generators[name]=rank_model.get_generator(eps,m=m)
        df_alpha = scores_for_generators(all_generators,scores=scores,runs=runs,details=details,latex_names=latex_names)
        df_alpha["epsilon"]=[eps]*len(df_alpha)
        all_dfs.append(df_alpha)
    all_alpha=pd.concat(all_dfs)
    all_alpha.reset_index(inplace=True,drop=True)
    
    if latex_names:
        all_alpha.rename({"epsilon":"$\\epsilon$"},axis=1,inplace=True)
    return all_alpha


def scores_for_rank_functions(rank_functions,n,m,scores=None,epsilons=0,runs=1,latex_names=True):
    """Scores for a list of rank functions

    Args:
        rank_functions (_type_): A dictionary of rank functions such as {name:rank_function}
        n (_type_): number of nodes
        m (_type_): number of edges
        scores (_type_, optional): A dictionary of scores such as {name:score}. Defaults to None.
        epsilons (int, optional): A list of epsilons. Defaults to 0.
        runs (int, optional): number of runs. Defaults to 1.
        latex_names (bool, optional):   If True, the names of the scores are latex formulas. Defaults to True.

    Returns:
        _type_: dataframe with the scores
    """
    rank_models = {name:stn.Rank_model(structure_function(n)) for name,structure_function in rank_functions.items()}
    return scores_for_rank_models(rank_models,m=m,scores=scores,epsilons=epsilons,runs=runs,latex_names=latex_names)
    # all_generators={}
    # if isinstance(epsilons,int):
    #     epsilons=[epsilons]
    # for i,(name,structure_function) in enumerate(rank_functions.items()):
    #     all_generators[name]=structure_function(n)
    # return scores_for_rank_models(all_generators,scores=scores,epsilons=epsilons,m=m)

def compare_graphs(df_reference,df_graphs,best_by_name=False,score_difference=False):
    """Compares a list of graphs to a reference graph
    
    Returns a dataframe with the scores of the graphs and the difference to the reference graph

    Args:
        df_reference (_type_): The scores of the reference graph as a dataframe
        df_graphs (_type_): The scores of the graphs to compare as a dataframe
        best_by_name (bool, optional): Returns the best graph for each name. Defaults to False.
        score_difference (bool, optional): Returns the difference to the reference graph instead of the scores. Defaults to False.

    Returns:
        _type_: A dataframe with the scores of the graphs and the difference to the reference graph
    """

    score_differences= []
    all_scores=df_reference.columns[1:]
    scores_reference = df_reference[all_scores].iloc[0]
    additonal_columns = [col for col in df_graphs.columns if col not in all_scores]
    df_graphs=df_graphs.copy().reset_index(drop=True)
    for i,row in df_graphs.iterrows():
        score_differences.append([1-abs(row[score]-scores_reference[score]) for score in all_scores])

    df_diff_scores=pd.DataFrame.from_records(score_differences,columns=list(all_scores))
    df_diff_scores["distance"]=len(all_scores)-df_diff_scores.sum(axis=1)
    
    if score_difference==False:
        #return the original scores computed for the selected rows instead of the difference
        df_to_return=df_graphs.copy()

        df_to_return["distance"]=list(df_diff_scores["distance"])
    else:
        df_to_return=pd.merge(df_diff_scores,df_graphs[additonal_columns],left_index=True,right_index=True)
        #   print(df_to_return)
        
    if best_by_name:
        df_to_return=df_to_return.sort_values("distance").groupby("name").first().reset_index()
        
    df_to_return=df_to_return.sort_values("distance")
    df_to_return=df_to_return.reset_index(drop=True)
    

        
    return df_to_return

def get_default_scores(with_size=False,latex_names=True):
    """Returns the default scores
    
    Returns a dictionary of scores such as {name:score} 
    
    Args:
        with_size (bool, optional): If True, the size of the graph is added to the scores. Defaults to False.
        latex_names (bool, optional): If True, the names of the scores are latex formulas. Defaults to False.
    
    Returns:
        _type_: A dictionary of scores such as {name:score}
    
    """
    if with_size:
        return default_scores|size

    if latex_names:
        return _names2latex(default_scores)
    return default_scores

def _names2latex_list(scores):
    to_return = []
    for k in scores:
        if k in score_names:
            to_return.append(score_names[k])
        else:
            to_return.append(k)
    return to_return

def _names2latex(scores):
    to_return = {}
    for k,v in scores.items():
        if k in score_names:
            to_return[score_names[k]]=v
        else:
            to_return[k]=v
    return to_return

default_scores = {"transitivity":transitivity,"average_clustering":average_clustering,"coreness":coreness,"average_shortest_path_length":average_shortest_path_length,"robustness":robustness,"giant_component_ratio":giant_component_ratio,"modularity":modularity,"boundaries":boundaries,"degree_heterogeneity":degree_heterogeneity,"degree_assortativity":degree_assortativity,"hierarchy":hierarchy}

#default_scores = {"$CC(G)$":transitivity,"$\overline{(CC(u))}$":average_clustering,"Core":coreness,"$\overline{d}$":average_shortest_path_length,"Rob":robustness,
#            "I":giant_component_ratio,"$Q$":modularity,"$Q_{bound}$":boundaries,
#            "$\sigma(k)$":degree_heterogeneity,
#            "${-(k-k)}$":lambda x: degree_assortativity(x,disassortativity=True),"${\propto(k,CC)}$":hierarchy}

score_names={"transitivity":"$CC(G)$","average_clustering":"$\overline{CC(u)}$","coreness":"Core","average_shortest_path_length":"$\overline{d}$","robustness":"Rob","giant_component_ratio":"I","modularity":"$Q$","boundaries":"$Q_{bound}$","degree_heterogeneity":"$\sigma(k)$","degree_assortativity":"$$-(k \propto k)$$","hierarchy":"$${k \propto CC}$$","nb_nodes":"$n$","nb_edges":"$m$","epsilon":"$\epsilon$"}

size={"nb_nodes":nx.number_of_nodes,"nb_edges":nx.number_of_edges}