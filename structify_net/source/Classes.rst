Classes
=========================================


Introduction 
-----------------------------------------

.. currentmodule:: structify_net


There are two classes allowing to represent network structures.

- :class:`Rank_model` is used to represent a network structure with a given number of nodes, and define the ranking of its node pairs, but is independent on the desired number of edges and of the function used to define the probability to observe an edge between two nodes given their rank.
- :class:`Graph_generator` is a model that can directly generate networks with a given number of nodes and expected number of edges. An instance can be created from `:class:`Rank_model`.

Details of the classes
-----------------------------------------

.. autoclass:: Rank_model
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Graph_generator
   :members:
   :undoc-members:
   :show-inheritance: