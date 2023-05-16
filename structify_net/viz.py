import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import itertools

def _plot_matrix(dict_values,node_order,ax=None,**kwargs):
    """Plot a matrix

    Args:
        matrix (_type_): a matrix
        ax (_type_, optional): a matplotlib axis. Defaults to None.

    Returns:
        _type_: a matplotlib plot
    """
    n=len(node_order)
    matrix = np.zeros((n,n))
    for i,j in itertools.combinations(range(n),2):
        matrix[i,j]=dict_values[frozenset((node_order[i],node_order[j]))]
        matrix[j,i]=dict_values[frozenset((node_order[i],node_order[j]))]
            
    #if node_order!=None:
    #    matrix=matrix[node_order,:]
    #    matrix=matrix[:,node_order]

    heatmap_args={'cmap':"YlGnBu_r",'cbar':False,'xticklabels':False,'yticklabels':False}
    for k,v in kwargs.items():
        heatmap_args[k]=v    
    m = sns.heatmap(matrix,ax=ax,**heatmap_args)

    return m

def _plot_rank_matrix(rank_model,nodeOrder=None,ax=None,**kwargs):
    """
    Plot a matrix of the graph, ordered by nodePair_order
    graph: a networkx graph
    nodePair_order: a list of node pairs, from the most likely to the less likely
    nodeOrder: a list of nodes, ordered by the order in which they should appear in the matrix
    """
    nodePair_order=rank_model.sortedPairs
    if nodeOrder!=None:
        n=len(nodeOrder)
    else:
        n=len(set([n for e in nodePair_order for n in e]))


    matrix = np.zeros((n,n))
    for i,e in enumerate(nodePair_order):
        e_l=list(e)
        matrix[e_l[0],e_l[1]]=i
        matrix[e_l[1],e_l[0]]=i

    if nodeOrder!=None:
        matrix=matrix[nodeOrder,:]
        matrix=matrix[:,nodeOrder]

    heatmap_args={'cmap':"YlGnBu_r",'cbar':False,'xticklabels':False,'yticklabels':False}
    for k,v in kwargs.items():
        heatmap_args[k]=v
        
    #m = sns.heatmap(matrix,cmap="YlGnBu_r",cbar=False,cbar_kws={'label': 'Rank'},xticklabels=False,yticklabels=False,ax=ax)
    m = sns.heatmap(matrix,ax=ax,**heatmap_args)

    return m

def plot_adjacency_matrix(g,nodeOrder=None):
    """Plot a matrix of the graph

    Parameters
        g: a networkx graph
        nodeOrder: a list of nodes, ordered by the order in which they should appear in the matrix
        
    """
    n=len(g.nodes)
    if nodeOrder==None:
        nodeOrder=g.nodes

    matrix = np.zeros((n,n))
    for e in g.edges:
        matrix[e[0],e[1]]=1
        matrix[e[1],e[0]]=1
        
    if nodeOrder!=None:
        matrix=matrix[nodeOrder,:]
        matrix=matrix[:,nodeOrder]

    m = sns.heatmap(matrix,cmap="binary",cbar=False)

    return m

def _plot_proba_function(generator,cumulative=False,ax=None):

    probas = generator.probas
    sorted_probas = sorted(probas,reverse=True)
    #res = sns.lineplot(x=range(len(probas)),y=sorted(probas,reverse=True))
    if cumulative:
        sorted_probas=np.cumsum(sorted_probas)
    if ax==None:
        fig,ax=plt.subplots()
    res = ax.plot(range(len(probas)),sorted_probas)

    if cumulative:
        plt.ylabel("Expected number of edges")
    else:
        plt.ylabel("Probability to observe an edge")
    plt.xlabel("Position of sorted node pairs")
    return res 


def _get_palette(nb_colors):
    """
    Get a palette of nb_colors colors
    """
    _my_pallete = sns.color_palette("husl", nb_colors)
    return _my_pallete

def spider_plot(df,categories=None,reference=None):
    """Plot a spider plot for each row of df

    This plot is used to compare the performance of different models on different scoring functions.
    
    Args:
        df (_type_): The dataframe to plot. Each row is a spider plot. The first column is the name of the spider plot. The other columns are the scoring functions to plot.
        categories (_type_, optional): A list of the scoring functions to plot. Defaults to None, in which case all the columns are plotted.
        reference (_type_, optional): Which line to use as a reference. Defaults to None, in which case no reference is plotted
    """
    each_spider=(340,360)
    df=df.copy()
    if "epsilon" in df.columns:
        df.rename(columns={"epsilon":"$\epsilon$"},inplace=True)
    if "$\epsilon$" in df.columns:
        df["name"]=df["name"]+"($\epsilon$="+df["$\epsilon$"].round(4).astype(str)+")"
        df.drop(columns=["$\epsilon$"],inplace=True)
    df.reset_index(inplace=True)
    names=list(df["name"])
    
    df.drop(columns=["index","name"],inplace=True) 
    if categories==None:
        categories=list(df)
    
    
    if isinstance(reference,int):
        reference=df.loc[reference][categories].values.flatten().tolist()
            
    my_dpi=96
    if len(df)>3:
        nb_rows=math.ceil(len(df)/3)
        nb_cols=3
        fig_size=(each_spider[0]*nb_cols/my_dpi, each_spider[1]*nb_rows/my_dpi)
        fig, axs = plt.subplots(nb_rows,nb_cols, subplot_kw=dict(projection='polar'),figsize=fig_size, dpi=my_dpi)
    else:
        fig_size=(each_spider[0]*len(df)/my_dpi, each_spider[1]/my_dpi)
        fig, axs = plt.subplots(1,len(df), subplot_kw=dict(projection='polar'),figsize=fig_size, dpi=my_dpi)
        if len(df)==1:
            axs=np.array([axs])
    axs_flatten=axs.flatten()
    colors= _get_palette(len(df.index))
    
    for row in range(0, len(df.index)):
        values=df.loc[row][categories].values.flatten().tolist()
        p = _make_spider(values, categories, names[row], colors[row],ax=axs_flatten[row],reference=reference) #, df[title][row]
        #axs_flatten[row].)
    if len(df)>3:
        to_delete=axs_flatten[len(df.index):]
        for ax in to_delete:
            fig.delaxes(ax)

pi=3.14159

def _model_evolution(df):
    df=df.copy()
    df.reset_index(inplace=True)
    df.drop(columns=["index"],inplace=True) 
    colors = sns.color_palette("Blues", len(df))
    for i in df.index:
        _make_spider(df.loc[i].values, df.columns, color=colors[i],fill=False)

def _make_spider(values,  categories, title="", color="green",fill=True,ax=None,reference=None):

    # number of variable
    #categories=list(df)[1:]
    categories=categories.copy()
    categories= [cat.replace("$$", "$") for cat in categories]

    N = len(categories)
    if isinstance(values,np.ndarray):
        values=values.tolist()

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles.append(angles[0]) #to close the line
    values=values.copy()
    values.append(values[0]) #to close the line

    # Initialise the spider plot
    #ax = plt.subplot(4,4,row+1, polar=True, )
    if ax==None:
        ax = plt.subplot(111, polar=True,)

    
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    #plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_xticks(angles[:-1], categories, color='blue', size=10)
    # Draw ylabels
    ax.set_rlabel_position(10)
    ticks=list(np.linspace(0,1,5))
    
    ax.set_rticks(ticks,[str(v) for v in ticks], color="grey", size=7)
    ax.grid(True)



    plt.gcf().canvas.draw()
    #angles = np.linspace(0,2*np.pi,len(ax.get_xticklabels())+1)
    #angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
    #angles = [a if math.sin(a)>0 else pi-a for a in angles ]
    
    angles_labels = np.rad2deg(angles)
    angles_labels = [360-a for a in angles_labels ]
    angles_labels = [180+a if 90<a<270 else a for a in angles_labels ]
    #angles = [-a for a in angles]
    labels = []
    for label, angle in zip(ax.get_xticklabels(), angles_labels):
        x,y = label.get_position()
        lab = ax.text(x,y+0.1, label.get_text(), transform=label.get_transform(),
                    ha=label.get_ha(), va=label.get_va(),color="grey", size=8)
        lab.set_rotation(angle)
        labels.append(lab)
    ax.set_xticklabels([])


    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    if reference!=None:
        reference=reference.copy()
        reference.append(reference[0]) #to close the line
        ax.fill(angles, reference, color="lightgrey", alpha=0.8)
        
    if fill:
        ax.fill(angles, values, color=color, alpha=0.0)
    
    # ax.tick_params(
    #     axis='both',
    #     #labelrotation=-45.,
    #     pad=15,
    #     )
    
    #ax.set_xtick_params(pad=10)
    ax.set_rmax(1)
    ax.set_rmin(0)
    # Add a title
    ax.set_title(title,pad=20)#, size=10, color=color)#, y=1.1)
    return(ax)



