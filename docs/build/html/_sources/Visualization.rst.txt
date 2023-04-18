Visualization
=========================================

The Structify_net library use mainly two types of visualizations: matrices and spider-plots.

Visualizations of the matrix plot of a particular model can be done by calling the corresponding functions from the corresponding classes:

.. currentmodule:: structify_net


- :func:`Rank_model::plot_matrix` for the matrix plot of a rank models
- :func:`Graph_generator::plot_proba_function` for the matrix plot of a Graph ER_generator

The visualization submodule (:mod:`structify_net.viz`) contains also functions to plot adjacency matrices and spider-plots.

Individual models
-----------------

.. autosummary::
   :toctree: generated/

    viz.plot_adjacency_matrix
    viz.spider_plot