Structure Zoo
==================

.. currentmodule:: structify_net

.. The structure zoo is a collection of structures. It contains both classic ones such as bloc structures, spatial structures or Watts-Strogatz structures, and less common ones. The zoo is a good place to start if you want to learn about the different types of structures that can be created with the library.

.. The scoring function are defined in the `zoo` submodule.
.. There are some useful shortcuts to get collection of structures from the 

.. - :data:`zoo.all_models_no_param`: contains all the model functions that do not require any parameter.
.. - :data:`zoo.all_models_with_m`: returns all the model functions that require a parameter `m` (Expected number of edges).
.. - :data:`zoo.all_models`: returns all the models functions

.. The function :func:`zoo.get_all_rank_models` return instanciated models. See details below.


Individual models
-----------------

.. autosummary::
   :toctree: generated/

   zoo.sort_distances
..    zoo.sort_blocks_assortative
..    zoo.sort_overlap_communities
..    zoo.sort_largest_disconnected_cliques
..    zoo.sort_stars
..    zoo.sort_core_distance
..    zoo.sort_spatial_WS
..    zoo.sort_fractal_leaves
..    zoo.sort_fractal_root
..    zoo.sort_nestedness
..    zoo.sort_fractal_hierarchical
..    zoo.sort_fractal_star
   
.. function to get instanciated model collections
.. ----------------------------------------------
..  .. autosummary::
..    :toctree: generated/

..    zoo.get_all_rank_models