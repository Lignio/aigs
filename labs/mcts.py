# imports
from __future__ import annotations
import numpy as np
import aigs
from aigs import State, Env
from dataclasses import dataclass, field


# %% Setup
env: Env


def heuristic_value(board: np.ndarray) -> int:
    return np.sum(board)


# %%
def minimax(state: State, maxim: bool, depth: int) -> int:
    if state.ended or depth > 10:
        return state.point if state.ended else heuristic_value(state.board)
    else:
        temp: int = -10 if maxim else 10
        for action in np.where(state.legal)[0]:  # for all legal actions
            value = minimax(env.step(state, action), not maxim, depth + 1)
            temp = max(temp, value) if maxim else min(temp, value)
        return temp


def alpha_beta(state: State, maxim: bool, alpha: int, beta: int) -> int:
    raise NotImplementedError  # you do this


@dataclass
class Node:
    state: State  # Add more fields
    children: np.array #might not be needed? make list instead, prolly easier (might be slightly more expensive)
    untriedActions: list #list of tried actions, maybe use untried actions instead?
    action: int
    parent: Node


# Intuitive but difficult in terms of code
def monte_carlo(state: State, cfg) -> int:
    #raise NotImplementedError  # you do this
    #go through the tree using tree_policy
    root = Node(state, 0)
    #while x < y:
        #sample tree_policy



def tree_policy(node: Node, cfg) -> Node:


    raise NotImplementedError  # you do this
    while node.terminal is false:
        if node.untriedActions.size > 0:
            return expand (node)
        else:
            return bestchild
        
    #Explore vs Exploit, something like 70/30 in favor of exploiting?


def expand(v: Node) -> Node:
    raise NotImplementedError  # you do this
    #add new node to tree
    #select action
    action = np.random(v.untriedActions)
    newState = Env.step(v.state, action)
    v.untriedAction.remove(action)
    child = Node(newState)
    child.action = action
    child.untriedActions = newState.legal
    child.Parent = v
    v.children.add(child)
   


def best_child(root: Node, c) -> Node:
    raise NotImplementedError  # you do this
    #define (current) best child node of the tree


def default_policy(state: State) -> int:
    raise NotImplementedError  # you do this
    #policy used by default to traverse the tree


def backup(node, delta) -> None:
    raise NotImplementedError  # you do this
    #set the default policy of the previous node?


# Main function
def main(cfg) -> None:
    global env
    env = aigs.make(cfg.game)
    state = env.init()

    while not state.ended:
        actions = np.where(state.legal)[0]  # the actions to choose from

        match getattr(cfg, state.player):
            case "random":
                a = np.random.choice(actions).item()

            case "human":
                print(state, end="\n\n")
                a = int(input(f"Place your piece ({'x' if state.minim else 'o'}): "))

            case "minimax":
                values = [minimax(env.step(state, a), not state.maxim) for a in actions]
                a = actions[np.argmax(values) if state.maxim else np.argmin(values)]

            case "alpha_beta":
                values = [alpha_beta(env.step(state, a), not state.maxim, -1, 1) for a in actions]
                a = actions[np.argmax(values) if state.maxim else np.argmin(values)]

            case "monte_carlo":
                raise NotImplementedError

            case _:
                raise ValueError(f"Unknown player {state.player}")

        state = env.step(state, a)

    print(f"{['nobody', 'o', 'x'][state.point]} won", state, sep="\n")
