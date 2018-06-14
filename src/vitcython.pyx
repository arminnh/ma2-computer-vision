import numpy as np
from numba import jit

def _getNextPos(prev_layer, y, maxY, r=1):
    prev = []
    if y+1 < maxY:
        prev.append((y+1, prev_layer[y+1]))

    prev.append((y, prev_layer[y]))

    if y > 0:
        prev.append((y-1, prev_layer[y-1]))

    return prev

def getMin(values):
    current = None
    bestVal = None
    for v in values:
        if bestVal is None:
            bestVal = v[1]
            current = v
        else:
            if v[1] < bestVal:
                current = v
                bestVal = v[1]

    return current

def viterbi_best_path(matrix):
    """Compute a path through a trellis graph minimizing the path cost.
    Returns:
      List of state indices: one state for each timestamp.
    """

    # List of lists: each list is a list of indices of the best previous
    # state for the given current state. That is, this is structured like
    # trellis, but stores pointers (indices) to previous states instead of
    # state representations.
    trellis_best_parents = []
    trellis = matrix.T
    # As above, but for total path cost to each point, not parent indices.
    trellis_path_costs = []

    # Iterate left-to-right through layers of the graph.
    prev_layer = []
    prev_total_costs = [0]
    for x in range(len(trellis)/4):
        layer = trellis[4*x]
        #print("Layer {}/{}".format(x, len(trellis)))
        total_costs = []
        best_parents = []

        for y, state in enumerate(layer):
            # List of total costs of getting here from each previous state.
            possible_path_costs = []
            if len(prev_layer) > 0:
                # Compute the cost of each possible transition to this
                # state, and combine with the total path cost of getting to
                # the previous state.
                for i, prev_state in _getNextPos(prev_layer, y, len(matrix)):
                    transition_cost = state
                    possible_path_costs.append(((4*x, i),
                                                transition_cost + prev_total_costs[i]))
            else:
                # No previous layer. Just store a cost of 0.  Essentially
                # this creates a common "START" node, idx 0, that all
                # first-layer nodes connect to.
                possible_path_costs.append(((0,y), state))

            # Find the min and argmin.
            (argmin_idx, min_val) = getMin(possible_path_costs)

            # Compute the cost of being in this state.
            state_cost = 1

            # Record best parent and best-path cost at this node.
            best_parents.append(argmin_idx)
            total_costs.append(min_val + state_cost)

        # Record the best parent for each node in this layer.
        trellis_best_parents.append(best_parents)
        trellis_path_costs.append(total_costs)
        prev_layer = layer
        prev_total_costs = total_costs

    # Done with forward pass.

    # Go backwards to reconstruct the best path to the end.
    best_states = []  # Build this up backwards and then reverse at end.

    # Find the best state in the last layer.
    next_state = (-1, np.argmin(trellis_path_costs[-1]))
    #best_states.append(next_state)

    trellis_best_parents.reverse()
    for parents in trellis_best_parents:
        next_state = parents[next_state[1]]
        best_states.append(next_state)

    return list(reversed(best_states))