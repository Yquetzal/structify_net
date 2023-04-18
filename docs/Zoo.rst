Structure Zoo
==================




The structure zoo is a collection of structures. It contains both classic ones such as bloc structures, spatial structures or Watts-Strogatz structures, and less common ones. The zoo is a good place to start if you want to learn about the different types of structures that can be created with the library.

The scoring function are defined in the :mod:`structify_net.zoo` submodule.
There are some useful shortcuts to get collection of structures from the 

- :data:`all_models_no_param`: contains all the model functions that do not require any parameter.
- :data:`all_models_with_m`: returns all the model functions that require a parameter `m` (Expected number of edges).
- :data:`all_models`: returns all the models functions

The function :func:`get_all_rank_models` return instanciated models. See details below.

.. currentmodule:: structify_net.zoo

Individual models
-----------------

.. autosummary::
   :toctree: generated/

   sort_distances
   sort_blocks_assortative
   sort_overlap_communities
   sort_largest_disconnected_cliques
   sort_stars
   sort_core_distance
   sort_spatial_WS
   sort_fractal_leaves
   sort_fractal_root
   sort_nestedness
   sort_fractal_hierarchical
   sort_fractal_star
   
function to get instanciated model collections
----------------------------------------------
 .. autosummary::
   :toctree: generated/

   get_all_rank_models