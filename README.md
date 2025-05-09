### A hobby project to learn reinforcement learning
Where I attempt to build and train a CNN from scratch such that it can play (well) a simple board game I made a while ago. The current approach is something like "Actor-Critic TD($\lambda $) Policy Gradient in episodic self-play with separate eligibility traces tracking the separate trajectories of two players in an episode". 

It may seem strange that square grid kernels are used in a board game with hexagonal tiles, but it works out as cube coordinates for a hexagonal grid are a 2D plane (see https://www.redblobgames.com/grids/hexagons/#coordinates). And topological features are preserved, with:

Roughly speaking, in a grid represented by (*q*, *r*) a 3x3 kernel sees:

|   |   |   |
|---|---|---|
| ❌ | ➕ | ➕ |
| ➕ | 🔵 | ➕ |
| ➕ | ➕ | ❌ |
|   |   |   |

With plus signs corresponding to the six tiles adjacent to a hexagon. The X's are at a distance of 2 from the central tile. Only 3x3 kernels are used in the architecture, and it is likely that the network can learn on its own to take into account the asymmetry in representation described above.
