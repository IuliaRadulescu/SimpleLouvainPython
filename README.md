# SimpleLouvainPython

## A simple, pure-python, efficient Louvain implementation

### How to use:

* Import the code: import louvainEfficient
* Create the instance: louvainEfficientInstance = louvainEfficient.LouvainEfficient()
* Call the main method: louvainEfficientInstance.louvain(graphAdjMatrix)
* The only param you have to provide is the graph adjancey matrix as a numpy 2-d array i.e. shape: (20, 20)
* The method returns a node2community mapping (nodeId - communityId)

## Dependencies

* Numpy is the only dependency - you probably already installed it since you're looking for Louvain implementations

## Why another Louvain implementation?

* It's pure python - easy to understand and extend
* It's quite efficient - due to numpy
* Easy to combine with igraph or networkx

## Hope you enjoy!
