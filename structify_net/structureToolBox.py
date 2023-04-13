import statistics
import math
import networkx as nx
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
from .rank2proba import Rank_model, Graph_generator
import numbers




def _largest_component(graph):
    Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
    G0 = graph.subgraph(Gcc[0])
    return G0

def has_giant_component(graph,threshold=0.9):
    return giant_component_ratio(graph)>threshold
        

def giant_component_ratio(graph):
    nb_nodes=graph.number_of_nodes()
    largest_CC= len(max(nx.connected_components(graph), key=len))
    ratio_CC=largest_CC/nb_nodes
    return ratio_CC

def transitivity(graph):
    return nx.transitivity(graph)

def average_clustering(graph):
    #print(graph.clustering())
    return np.average([cc for n,cc in  nx.clustering(graph).items()])

def average_shortest_path_length(graph,normalized=True):
    if has_giant_component(graph):
        if giant_component_ratio(graph)<1:
            graph=_largest_component(graph)
        
        graph_shortest=nx.average_shortest_path_length(graph)
        
       
        if normalized:
            n= graph.number_of_nodes()
            avg_degree=graph.number_of_edges()/n*2
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
    mod=nx.algorithms.community.modularity(graph,nx.algorithms.community.louvain_communities(graph))
    if normalized:
        ref_model = Graph_generator.ER(graph.number_of_nodes(),nx.density(graph))
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
    degrees = [d for n, d in graph.degree()]
    heterogeneity = _gini_coefficient(degrees)
    return heterogeneity
    #avg_degree=statistics.mean([d for n, d in graph.degree()]) 
    #degree_heterogeneity=statistics.stdev(degrees)-statistics.sqrt(avg_degree)

def is_degree_heterogeneous(graph):
    avg_degree = np.average([d for n, d in graph.degree()])
    reference_threshold = _gini_coefficient(np.random.poisson(avg_degree,len(graph.nodes)))
    power_law_threshold = _gini_coefficient(nx.utils.powerlaw_sequence(len(graph.nodes),3))
    #print("-",degree_heterogeneity(graph),reference_threshold,power_law_threshold)
    return degree_heterogeneity(graph)>((reference_threshold+power_law_threshold)/2)

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
    return _robustness_func(graph)

def degree_assortativity(graph,normalized=False,positive_only=True,disassortativity=False):
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
                hierarchical=-spearmanr(degrees,clusterings).correlation
                break
            
        if np.isnan(hierarchical):
            hierarchical=0
    if normalized:
        hierarchical=hierarchical/2+0.5
    if positive_only:
        hierarchical=max(0,hierarchical)
    return hierarchical

def boundaries(graph,normalized=True):
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
    coreness = max(list(nx.core_number(graph).values()))
    
    #max_possible_coreness = math.floor(1/2 * (math.sqrt( 8*graph.number_of_edges())+1))
    if normalized:
        max_possible_coreness = math.floor(math.sqrt(graph.number_of_edges()))
        coreness= coreness/max_possible_coreness
    return coreness

def compute_all_scores(graph):
    to_return = {}
    for name,func in all_scores.items():
        to_return[name]=func(graph)
    return to_return

def scores_for_graphs(graphs,scores=None):
    records=[]
    if scores is None:
        scores = get_default_scores()
    for graph_name,graph in graphs.items():
        line=[]
        for score_name,func in scores.items():
            line.append(func(graph))
        records.append([graph_name]+line)
    return pd.DataFrame(records,columns=["name"]+list(scores.keys()))

def scores_for_generators(generators,scores=None,runs=1,details=False):
    to_return = pd.DataFrame()
    for i in range(runs):
        graphs = {name:generator.generate() for name,generator in generators.items()}
        results= scores_for_graphs(graphs,scores=scores)
        to_return = pd.concat([to_return,results])
        #scores["run"]=[i]*len(scores)
    if details:
        return to_return
    to_return=to_return.groupby("name").mean().reset_index()
    return to_return


def scores_for_rank_models(rank_models,m,scores=None,epsilons=0,runs=1,details=False):
    all_generators={}
    if not isinstance(rank_models,dict):
        rank_models = {"model":rank_models}
    if isinstance(epsilons,numbers.Number):
        if epsilons<=1:
            epsilons=[epsilons]
        else:
            epsilons=[0]+list(np.logspace(-4,0,epsilons-1))
    all_dfs=[]
    for eps in epsilons:
        for i,(name,rank_model) in enumerate(rank_models.items()):
            all_generators[name]=rank_model.get_generator(eps,m=m)
        df_alpha = scores_for_generators(all_generators,scores=scores,runs=runs,details=details)
        df_alpha["epsilon"]=[eps]*len(df_alpha)
        all_dfs.append(df_alpha)
    all_alpha=pd.concat(all_dfs)
    return all_alpha


def scores_for_rank_functions(rank_functions,n,m,scores=None,epsilons=0,runs=1):
    rank_models = {name:Rank_model(structure_function(n)) for name,structure_function in rank_functions.items()}
    return scores_for_rank_models(rank_models,m=m,scores=scores,epsilons=epsilons,runs=runs)
    # all_generators={}
    # if isinstance(epsilons,int):
    #     epsilons=[epsilons]
    # for i,(name,structure_function) in enumerate(rank_functions.items()):
    #     all_generators[name]=structure_function(n)
    # return scores_for_rank_models(all_generators,scores=scores,epsilons=epsilons,m=m)

def compare_graphs(df_reference,df_graphs,best_by_name=False,score_difference=False):

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

def get_default_scores(with_size=False):
    if with_size:
        return default_scores|size
    return default_scores


default_scores = {"$CC(G)$":transitivity,"$\overline{(CC(u))}$":average_clustering,"Core":coreness,"$\overline{d}$":average_shortest_path_length,"Rob":robustness,
            "I":giant_component_ratio,"$Q$":modularity,"$Q_{bound}$":boundaries,
            "$\sigma(k)$":degree_heterogeneity,
            "$-\propto(k-k)$":lambda x: degree_assortativity(x,disassortativity=True),"$\propto(k,CC)$":hierarchy}
size={"nb_nodes":nx.number_of_nodes,"nb_edges":nx.number_of_edges}