Scoring
==================

.. currentmodule:: structify_net


The scoring submodule contains a collection of scoring function to describe graphs. The scoring function are used to compare graphs.

The scoring function are defined in the :mod:`scoring` module.

- :data:`scoring.default_scores`: contains all available scores in a dictionary {name: score}.
- :data:`scoring.size`: contains additional scores describing the size of the graphs (number of nodes, number of edges)
- :data:`scoring.score_names`: contains a dictionary to convert plain score names to short latex names.

The function :func:`scoring.get_default_scores` return the default_scores in a convenient way, see below



Individual scoring functions
----------------------------

.. autosummary::
   :toctree: generated/

   scoring.has_giant_component
   scoring.giant_component_ratio
   scoring.transitivity


Useful functions
----------------

.. autosummary::
   :toctree: generated/
   
   scoring.compute_all_scores
   scoring.scores_for_graphs
   scoring.scores_for_generators
   scoring.scores_for_rank_models
   scoring.scores_for_rank_functions
   scoring.compare_graphs
   scoring.get_default_scores