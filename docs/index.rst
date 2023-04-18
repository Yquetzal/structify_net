.. Structify_net documentation master file, created by
   sphinx-quickstart on Mon Apr 17 09:48:51 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Structify_net's documentation!
=========================================

What is Structify_net?
----------------------

Structify_net is a python library allowing to create networks with a predefined structure, and a chosen number of nodes and links.

The principle is to use a common framework to encompass community/bloc structure, spatial structure, and many other types of structures. 

More specifically, a structure is defined by: 
- a number of nodes `n`
- a ranking of all the pairs of nodes, from most likely to be present to less likely to be present.

We can instanciate such a model using the `Rank_model` class.

This abstract structure can be instanciated into a graph generator by adding:
- An expected number of edges `m`
- a probability distribution assigning a probability of observing an edge to each rank
We can instanciate such a model using the `Graph_generator` class.


This library also contains a set of tools to visualize and score the resulting networks.

.. toctree::
   :maxdepth: 2

   self
   Classes
   Zoo
   Visualization   
   Scoring
   Tutorial/Tutorial


