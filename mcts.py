import sys
if sys.version_info.major == 2 or (sys.version_info.major == 3
                                   and sys.version_info.minor < 6):
    sys.exit('Please upgrade your python to version >= 3.6.0')

import random
import math

import glog
import ray
from tqdm import tqdm

from state import State


class Node:
    def __init__(self, state: State, parent=None):
        self.children = []  # list of (Node)s
        self.state = state  # list of (State)s - Must provide next_state() func
        self.parent = parent  # parent Node
        self.reward = 0.
        self.visits = 1

    def get_score(self, parent_visits: int, scalar: float) -> float:
        exploit = self.reward / self.visits
        explore = math.sqrt(2. * math.log(parent_visits) / float(self.visits))
        return exploit + scalar * explore

    def get_best_child(self, scalar: float = 0.7071067811865475) -> 'Node':
        """Return best child under current node. Score based on original MCTS
        formula.

           Default scalar = 1 / math.sqrt(2.)
        """
        best_score = 0.
        best_children = []
        for child in self.children:
            score = child.get_score(self.visits, scalar)
            if score == best_score:
                best_children.append(child)
            elif score > best_score:
                best_score = score
                best_children = [child]
        if len(best_children) == 0:
            glog.error("Fatal error: No best child found.")
        return random.choice(best_children)

    def add_child(self, child_state: State) -> None:
        child = Node(child_state, self)
        self.children.append(child)

    def expand(self) -> 'Node':
        tried_states = [child.state for child in self.children]
        new_state = self.state.next_state()
        while new_state in tried_states:
            new_state = self.state.next_state()
        self.add_child(new_state)
        return self.children[-1]

    def fully_expanded(self) -> bool:
        return len(self.children) == State.MAX_MOVE

    def simulate(self) -> float:
        state = self.state
        while not state.terminal():
            state = state.next_state()
        return state.reward()


@ray.remote
class MCTSAgent:
    def __init__(self, root: Node):
        self.root = root

    def get_root_state(self):
        return (self.root.reward, self.root.visits)

    def select_expand(self, node) -> Node:
        while not node.state.terminal():
            if len(node.children) == 0:
                return node.expand()
            elif node.fully_expanded() or random.uniform(0, 1) < .5:
                node = node.get_best_child()
            else:
                return node.expand()
        return node

    def back_propagate(self, node: Node, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def uct_search(self, budget: float) -> State:
        for iter in range(int(budget)):
            glog.info(f'ITER: {iter} / BUDGET: {int(budget)}')
            child = self.select_expand(self.root)
            reward = child.simulate()
            glog.info(f'reward after simulation: {reward}')
            self.back_propagate(child, reward)
        best_child = self.root.get_best_child(0)
        self.root = best_child
        return best_child.state


if __name__ == "__main__":
    ray.init()
    root = Node(State())
    mcts = MCTSAgent.remote(root)
    states = ray.get([mcts.uct_search.remote(10 / (l + 1)) for l in range(5)])
    glog.info(f'states: {states}')
    # for l in range(5):
    #     state = ray.get(mcts.uct_search.remote(10 / (l + 1)))
    #     glog.info(f'result state: {state}')
    #     root_state = ray.get(mcts.get_root_state.remote())
    #     glog.info(f'root.state: {root_state}')
