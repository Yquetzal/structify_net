.. code:: ipython3

    %load_ext autoreload
    %autoreload 2
    import networkx as nx
    import itertools
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    import netwulf
    import seaborn as sns
    
    import structify_net as stn
    import structify_net.viz as viz
    import structify_net.zoo as zoo
    import structify_net.scoring as scoring


.. parsed-literal::

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload



Introduction to Structify_Net
=============================

Structify_Net is a network generator provided as a python library.

It allows to generate networks with: \* A chosen number of nodes and
edges \* A chosen structure \* A constrolled amount of randomness

Step 1: Graph properties definition
-----------------------------------

We start by defining the number of nodes and edges that we want

.. code:: ipython3

    n=128
    m=512

Step 2: Structure definion
--------------------------

We start by defining a structure, by ordering the pairs of nodes in the
graph from the most likely to appear to the less likely to appear. For
instance, if we assume that our network is a spatial network, and that
each node has a position in an euclidean space, we can define that the
pairs of nodes are ranked according to their distance in this space.

Many classic structures are already implemented in Structify-Net. For
the sake of example, here we define a very simple organisation, which
actually correspond to a nested structure, by defining a sorting
function. This simple function only requires the nodes ids. we could
provide node attributes in the third parameter.

.. code:: ipython3

    def R_nestedness(u,v,_):
        return u+v

We then generate a Rank_model object using Structify-net

.. code:: ipython3

    rank_nested = stn.Rank_model(n,R_nestedness)

A way to visualize the resulting structure is to plot the node pairs
order as a matrix

.. code:: ipython3

    figure(figsize=(4, 4), dpi=80)
    
    rank_nested.plot_matrix()



.. image:: output_10_0.png


Step 3: Edge probability definition
-----------------------------------

Now that we know which node pairs are the most likely to appear, we need
to define a function :math:`f` that assign edge probabilities on each
node pair, by respecting some constraints: \* The expected number of
edges must be equal to the chosen parameter ``m``,
i.e. :math:`\sum_{u,v\in G}f(rank(u,v))=m` \* For any two node pairs
:math:`e_1` and :math:`e_2`, if :math:`rank(e_1)>rank(e_2)`, then
:math:`f(e_1)\geq f(e_2)`

Although any such function can be provided, Structify-net provides a
convenient function generator, using a constraint parameter
:math:`\epsilon \in [0,1]`, such as 0 corresponds to a deterministic
structure, the m pairs of highest rank being connected by an edge, while
1 corresponds to a fully random network.

.. code:: ipython3

    probas = rank_nested.get_generator(epsilon=0.5,m=m)

We can plot the probability as a function of rank for various values of
``epsilon``

.. code:: ipython3

    fig, ax = plt.subplots()
    for epsilon in np.arange(0,1.1,1/6):
        probas = rank_nested.get_generator(epsilon=epsilon,m=m)
        elt = probas.plot_proba_function(ax=ax)
        #elt=viz.plot_proba_function(probas,ax=ax)
        elt[-1].set_label(format(epsilon, '.2f'))
        #fig_tem.plot(label="pouet"+str(epsilon))
    ax.legend(title="$\epsilon$")





.. parsed-literal::

    <matplotlib.legend.Legend at 0x2e8157d10>




.. image:: output_14_1.png


Step 4: Generate a graph from edge probabilities
------------------------------------------------

.. code:: ipython3

    generator = rank_nested.get_generator(epsilon=0.5,m=m)
    g_generated = generator.generate()

.. code:: ipython3

    figure(figsize=(4, 4), dpi=80)
    viz.plot_adjacency_matrix(g_generated)




.. parsed-literal::

    <AxesSubplot: >




.. image:: output_17_1.png


Whole process in a function
---------------------------

The whole process of graph generation from a desired number of nodes and
edges can be done in a single function

.. code:: ipython3

    g_example = rank_nested.generate_graph(epsilon=0,m=500)

Structure Zoo
-------------

StructifyNet already implement various graph structure. They are
exemplified below

.. code:: ipython3

    n=128
    m=512
    figure(figsize=(12, 12), dpi=80)
    for i,(name,rank_model) in enumerate(zoo.get_all_rank_models(n=128,m=m).items()):
        ax = plt.subplot(4,4,i+1 )
        
        rank_model.plot_matrix()
        
        ax.set_title(name)




.. image:: output_21_0.png


Graph description
-----------------

Models generate graphs having specific properties. Structify-Net
includes various scores that can be used to characterize networks –and
the model generating them.

Getting Started: replicating Watts-Strogatz Small World experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The famous small world experiments consisted in generating networks with
a locally clustered structure (nodes are located on a ring and connected
to their neighbors), and then introducing progressively random noise,
until reaching a random network. The *small world regime* corresponds to
the level of noise for which the clustering coefficient is still high
-as in the locally clustered network- but the average shortest path is
already low -as in a random network.

.. code:: ipython3

    n=1000
    m=n*5
    WS_model = zoo.sort_spatial_WS(n,k=10) #k is the nodes degree
    df_scores = WS_model.scores(m=m,
                    scores={"clustering":scoring.average_clustering,
                            "short paths":scoring.average_shortest_path_length},
                    epsilons=np.logspace(-4,0,10),latex_names=False)
    df_scores.head(3)



.. parsed-literal::

    Epsilon:   0%|          | 0/10 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>name</th>
          <th>clustering</th>
          <th>short paths</th>
          <th>epsilon</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>model</td>
          <td>0.664838</td>
          <td>0.043552</td>
          <td>0.000100</td>
        </tr>
        <tr>
          <th>1</th>
          <td>model</td>
          <td>0.663089</td>
          <td>0.055966</td>
          <td>0.000278</td>
        </tr>
        <tr>
          <th>2</th>
          <td>model</td>
          <td>0.657589</td>
          <td>0.092717</td>
          <td>0.000774</td>
        </tr>
      </tbody>
    </table>
    </div>



Plotting the results
~~~~~~~~~~~~~~~~~~~~

We can observe the small world regime by plotting the evolution of both
values as a function of ``epsilon``. Note that the ``short paths`` is
defined in a different way than in the original article, to be more
generic. It corresponds to the inverse of the average distance,
normalized such as ``short paths``\ =1 for a network with a complete
star, such as each node is at distance 2 from all other node.

.. code:: ipython3

    df_scores.plot(x="epsilon",logx=True,figsize=(8, 3))




.. parsed-literal::

    <AxesSubplot: xlabel='epsilon'>




.. image:: output_26_1.png


Small World regime for other structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can replicate this experiment for all structures in our structure zoo

.. code:: ipython3

    df_scores = scoring.scores_for_rank_models(zoo.get_all_rank_models(n,m),m=m,
                           scores={"clustering":scoring.average_clustering,
                                   "short paths":scoring.average_shortest_path_length},
                           epsilons=np.logspace(-4,0,10),latex_names=False)                  



.. parsed-literal::

    Epsilon:   0%|          | 0/10 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]


.. code:: ipython3

    df_scores.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>name</th>
          <th>clustering</th>
          <th>short paths</th>
          <th>epsilon</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ER</td>
          <td>0.009857</td>
          <td>0.444096</td>
          <td>0.0001</td>
        </tr>
        <tr>
          <th>1</th>
          <td>blocks_assortative</td>
          <td>0.329946</td>
          <td>0.000000</td>
          <td>0.0001</td>
        </tr>
        <tr>
          <th>2</th>
          <td>core_distance</td>
          <td>0.112838</td>
          <td>0.000000</td>
          <td>0.0001</td>
        </tr>
        <tr>
          <th>3</th>
          <td>disconnected_cliques</td>
          <td>0.978983</td>
          <td>0.000000</td>
          <td>0.0001</td>
        </tr>
        <tr>
          <th>4</th>
          <td>fractal_hierarchy</td>
          <td>0.829898</td>
          <td>0.971546</td>
          <td>0.0001</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    g = df_scores.groupby('name')
    
    fig, axes = plt.subplots(4,4, sharex=True)
    all_axes = axes.flatten()
    for i, (name, d) in enumerate(g):
        ax = d.plot.line(x='epsilon', ax=all_axes[i], title=name,logx=True,figsize=(10, 8))
        ax.set_ylim(-0.05,1.05)
        ax.legend().remove()




.. image:: output_30_0.png


Models Profiling
----------------

Average distance and Average clustering are only two examples of graph
structure descriptors. Structify-Net contains several other descriptors.
We can use them to show more details of the evolution from the regular
grid to the random network

.. code:: ipython3

    detail_evolution = zoo.sort_spatial_WS(500).scores(m=500*5,epsilons=np.logspace(-4,0,6),scores=scoring.get_default_scores(),latex_names=True)
    #detail_evolution = toolBox.scores_for_rank_functions({"spatialWS":zoo.sort_spatial_WS},500,500*5,epsilons=np.logspace(-4,0,6),scores=toolBox.get_all_scores())



.. parsed-literal::

    Epsilon:   0%|          | 0/6 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]


.. code:: ipython3

    detail_evolution




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>name</th>
          <th>$CC(G)$</th>
          <th>$\overline{CC(u)}$</th>
          <th>Core</th>
          <th>$\overline{d}$</th>
          <th>Rob</th>
          <th>I</th>
          <th>$Q$</th>
          <th>$Q_{bound}$</th>
          <th>$\sigma(k)$</th>
          <th>$$-(k \propto k)$$</th>
          <th>$${k \propto CC}$$</th>
          <th>$\epsilon$</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>model</td>
          <td>0.666075</td>
          <td>0.666182</td>
          <td>0.200000</td>
          <td>0.046832</td>
          <td>1.00</td>
          <td>1.0</td>
          <td>0.777533</td>
          <td>0.260278</td>
          <td>0.000398</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.000100</td>
        </tr>
        <tr>
          <th>1</th>
          <td>model</td>
          <td>0.656600</td>
          <td>0.657631</td>
          <td>0.183673</td>
          <td>0.108350</td>
          <td>0.98</td>
          <td>1.0</td>
          <td>0.773530</td>
          <td>0.224972</td>
          <td>0.008416</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.000631</td>
        </tr>
        <tr>
          <th>2</th>
          <td>model</td>
          <td>0.626598</td>
          <td>0.631821</td>
          <td>0.163265</td>
          <td>0.206054</td>
          <td>0.98</td>
          <td>1.0</td>
          <td>0.764524</td>
          <td>0.238881</td>
          <td>0.030597</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.003981</td>
        </tr>
        <tr>
          <th>3</th>
          <td>model</td>
          <td>0.447929</td>
          <td>0.466216</td>
          <td>0.142857</td>
          <td>0.355655</td>
          <td>0.98</td>
          <td>1.0</td>
          <td>0.654202</td>
          <td>0.217358</td>
          <td>0.077708</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.025119</td>
        </tr>
        <tr>
          <th>4</th>
          <td>model</td>
          <td>0.105552</td>
          <td>0.113990</td>
          <td>0.142857</td>
          <td>0.481359</td>
          <td>1.00</td>
          <td>1.0</td>
          <td>0.268974</td>
          <td>0.172912</td>
          <td>0.141030</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.158489</td>
        </tr>
        <tr>
          <th>5</th>
          <td>model</td>
          <td>0.017456</td>
          <td>0.016949</td>
          <td>0.140000</td>
          <td>0.518599</td>
          <td>0.98</td>
          <td>1.0</td>
          <td>0.021596</td>
          <td>0.070055</td>
          <td>0.163151</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.000000</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    viz.spider_plot(detail_evolution,reference=0)



.. image:: output_34_0.png


We can also compare properties of a set of models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case, we plot all models in Structify’s Zoo, with
``epsilon``\ =0

.. code:: ipython3

    n,m=128,128*8
    detail_evolution = scoring.scores_for_rank_models(zoo.get_all_rank_models(n,m),m,epsilons=0,latex_names=True)



.. parsed-literal::

    Epsilon:   0%|          | 0/1 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/1 [00:00<?, ?it/s]


.. code:: ipython3

    viz.spider_plot(detail_evolution,reference=0)



.. image:: output_37_0.png


Comparing with an observed network
----------------------------------

If we are interested in a particular network, we can compare the
structure of that network with the strucuture of some candidate models
in our strucuture zoo. For instance, let us check the structure of the
Zackary karate club graph

.. code:: ipython3

    karate_scores = scoring.scores_for_graphs({"karate club":nx.karate_club_graph()},latex_names=True)
    viz.spider_plot(karate_scores)



.. image:: output_39_0.png


Generate graphs of the same size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We generate graphs using the structures in the zoo, varying the epsilon
parameter, but keeping the same number of nodes and (expected) edges
than in the target graph. To get more reliable results, we take the
average values over multiple runs.

Since the karate club graph is often interpreted in term of communities,
we include two additional versions of the structures, that can be
parameterized with the number of blocks.

.. code:: ipython3

    n=nx.karate_club_graph().number_of_nodes()
    m=nx.karate_club_graph().number_of_edges()
    models_to_compare=zoo.get_all_rank_models(n,m)
    
    louvain_communities=nx.community.louvain_communities(nx.karate_club_graph())
    models_to_compare["louvain"]=zoo.sort_blocks_assortative(n,blocks=louvain_communities)
    models_to_compare["com=2"]=zoo.sort_blocks_assortative(n,blocks=2)
    epsilons=np.logspace(-3,0,10)


.. code:: ipython3

    compare_scores = scoring.scores_for_rank_models(models_to_compare,m,epsilons=epsilons,runs=20)



.. parsed-literal::

    Epsilon:   0%|          | 0/10 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/20 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/20 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/20 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/20 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/20 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/20 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/20 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/20 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/20 [00:00<?, ?it/s]



.. parsed-literal::

    Run:   0%|          | 0/20 [00:00<?, ?it/s]


Comparing
^^^^^^^^^

We compute the :math:`L_1` distance (sum of differences in each score)
between the observed graph and the models. We can explore how the
models’ similarity evolve as a function of the random parameter

.. code:: ipython3

    compare = scoring.compare_graphs(karate_scores,compare_scores,best_by_name=False,score_difference=True)

.. code:: ipython3

    ax = sns.lineplot(data=compare,x="$\epsilon$",y="distance",hue="name",style="name",palette=sns.color_palette("husl", len(models_to_compare)))
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.xscale('log')



.. image:: output_45_0.png


Details of models matching
--------------------------

We can study in more details what properties does each model captures or
not. We select for each model the value of epsilon giving the best
match. Models are also sorted according the distance, so that the first
models returned are the most similar, and then we plot the properties of
those selected models, with the properties of our graph for comparison

.. code:: ipython3

    compare = scoring.compare_graphs(karate_scores,compare_scores,best_by_name=True,score_difference=False)

.. code:: ipython3

    compare.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>name</th>
          <th>$CC(G)$</th>
          <th>$\overline{CC(u)}$</th>
          <th>Core</th>
          <th>$\overline{d}$</th>
          <th>Rob</th>
          <th>I</th>
          <th>$Q$</th>
          <th>$Q_{bound}$</th>
          <th>$\sigma(k)$</th>
          <th>$$-(k \propto k)$$</th>
          <th>$${k \propto CC}$$</th>
          <th>$\epsilon$</th>
          <th>distance</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>fractal_hierarchy</td>
          <td>0.280471</td>
          <td>0.564375</td>
          <td>0.461806</td>
          <td>0.869461</td>
          <td>0.254</td>
          <td>0.992647</td>
          <td>0.132835</td>
          <td>0.414314</td>
          <td>0.319198</td>
          <td>0.000000</td>
          <td>0.159376</td>
          <td>0.100000</td>
          <td>0.898688</td>
        </tr>
        <tr>
          <th>1</th>
          <td>fractal_root</td>
          <td>0.440271</td>
          <td>0.622941</td>
          <td>0.425000</td>
          <td>0.557126</td>
          <td>0.418</td>
          <td>0.998529</td>
          <td>0.302972</td>
          <td>0.476464</td>
          <td>0.279551</td>
          <td>0.043807</td>
          <td>0.681881</td>
          <td>0.100000</td>
          <td>1.023490</td>
        </tr>
        <tr>
          <th>2</th>
          <td>maximal_stars</td>
          <td>0.226890</td>
          <td>0.423235</td>
          <td>0.449306</td>
          <td>0.882993</td>
          <td>0.400</td>
          <td>0.970588</td>
          <td>0.000000</td>
          <td>0.285450</td>
          <td>0.439384</td>
          <td>0.000000</td>
          <td>0.070655</td>
          <td>0.464159</td>
          <td>1.198892</td>
        </tr>
        <tr>
          <th>3</th>
          <td>core_distance</td>
          <td>0.280774</td>
          <td>0.240385</td>
          <td>0.527778</td>
          <td>0.606878</td>
          <td>0.600</td>
          <td>0.957353</td>
          <td>0.007563</td>
          <td>0.369889</td>
          <td>0.362630</td>
          <td>0.039165</td>
          <td>0.004190</td>
          <td>0.464159</td>
          <td>1.481096</td>
        </tr>
        <tr>
          <th>4</th>
          <td>spatialWS</td>
          <td>0.314400</td>
          <td>0.282437</td>
          <td>0.500000</td>
          <td>0.547238</td>
          <td>0.600</td>
          <td>1.000000</td>
          <td>0.198066</td>
          <td>0.412887</td>
          <td>0.195659</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.001000</td>
          <td>1.498977</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    compare_plot=compare.drop(columns=["distance"])
    compare_plot.loc[-1, :] = karate_scores.iloc[0]
    compare_plot.sort_index(inplace=True) 

.. code:: ipython3

    viz.spider_plot(compare_plot,reference=0)



.. image:: output_50_0.png










