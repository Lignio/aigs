# imports
from __future__ import annotations
import numpy as np
import aigs
import pickle
from aigs import State, Env
from dataclasses import dataclass, field


# %% Setup
env: Env


# %%
def minimax(state: State, maxim: bool) -> int:
    if state.ended:
        return state.point
    else:
        temp: int = -10 if maxim else 10
        for action in np.where(state.legal)[0]:  # for all legal actions
            value = minimax(env.step(state, action), not maxim)
            temp = max(temp, value) if maxim else min(temp, value)
        return temp


def alpha_beta(state: State, maxim: bool, alpha: int, beta: int) -> int:
    raise NotImplementedError  # you do this


@dataclass
class Node:
    state: State  
    children: list 
    untriedActions: list 
    action: int
    parent: Node
    visitCount: int
    wins: int

    #Function for calculating uct, based on code in slides.
    def UCT(self, c)->float:
        val = (self.wins/self.visitCount) + c * np.sqrt(np.log(self.parent.visitCount/self.visitCount))
        return val


# Intuitive but difficult in terms of code
def monte_carlo(state: State, cfg) -> int:
    #create root node
    root = Node(state, [], np.where(state.legal)[0], 1, None, 1, 0)
    
    x = 0
    node = root
    #currently loop is setup to ensure that the tree always expands 10 layers down.
    while x < cfg.compute:
        if len(node.untriedActions) > 0:
            newNode = tree_policy(node, cfg)
            if newNode is None:
                break
            delta = default_policy(newNode.state)
            backup(newNode, delta)
        else:
            if x == 0:
                #update root node, to now include its children.
                root = node
            node = best_child(node, (1/np.sqrt(2)))
            x+=1
    return best_child(root, (1/np.sqrt(2))).action

""" Not used, was part of trying to implement saving the tree.
def traverse_tree(node: Node, state: State) -> Node:
        queue = []
        queue.append(node)
        while len(queue) > 0:
            print ("TestQ", len(queue))
            node = queue.pop()
            print ("TestC", len(node.children))
            for child in node.children:
                if child.state is state:
                    return child
                queue.append(child)
            
        return None
"""

#Decide whether to add new node or return best child.
def tree_policy(node: Node, cfg) -> Node:
    while node.state.ended is False:
        if len(node.untriedActions) > 0:
            return expand (node)
        else:
            return best_child(node, (1/np.sqrt(2)))

 #add new node to tree
def expand(v: Node) -> Node:
    action = np.random.choice(v.untriedActions).item()
    assert action in v.untriedActions
    newState = env.step(v.state, action)
    v.untriedActions = np.delete(v.untriedActions, np.where(v.untriedActions==action))
    child = Node(newState, [], np.where(newState.legal)[0], action, v, 0, 0)
    v.children.append(child)
    return child
   

#define (current) best child node of the tree
def best_child(root: Node, c) -> Node:
    bc = None
    val = 0
    for child in root.children:
        if child.UCT(c) > val:
            val = child.UCT(c)
            bc = child
    return bc
    

#Choose a random action from within the node.
def default_policy(state: State) -> int:
    while state.ended is False:
        action = np.random.choice(np.where(state.legal)[0]).item()
        nState = env.step(state, action)
        state = nState
    return state.point

#Update the tree.
def backup(node, delta) -> None:
    while node is not None:
        node.visitCount +=1
        node.state.point += delta
        delta = -delta
        node = node.parent


# Main function
def main(cfg) -> None:
    global env
    env = aigs.make(cfg.game)
    state = env.init()
    a = None

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
                a = monte_carlo(state, cfg)

            case _:
                raise ValueError(f"Unknown player {state.player}")
        state = env.step(state, a)

    print(f"{['nobody', 'o', 'x'][state.point]} won", state, sep="\n")
