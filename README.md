# structify-net
Structify_net is a python library allowing to create networks with a predefined structure, and a chosen number of nodes and links.

The principle is to use a common framework to encompass community/bloc structure, spatial structure, and many other types of structures.

More specifically, a structure is defined by:

* a number of nodes n
* a ranking of all the pairs of nodes, from most likely to be present to less likely to be present.


-----
# Insallation
You can install this library using pip:

```
pip install structify-net
```

You can then import it and its modules as:

```
import structify_net as stn
import structify_net.viz as viz
import structify_net.zoo as zoo
import structify_net.scoring as scoring
```

----

# Documentation
* The documentation of the library can be found at the following address: https://structify-net.readthedocs.io/en/latest/

* The method is described in a paper (yet to be published)