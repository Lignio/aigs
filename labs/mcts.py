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
    state: State  # Add more fields
    children: list #might not be needed? make list instead, prolly easier (might be slightly more expensive)
    untriedActions: list #list of tried actions, maybe use untried actions instead?
    action: int
    parent: Node
    visitCount: int
    wins: int

    def UCT(self, c)->float:
        val = (self.wins/self.visitCount) + c * np.sqrt(np.log(self.parent.visitCount/self.visitCount))
        return val


# Intuitive but difficult in terms of code
def monte_carlo(state: State, cfg) -> int:
    #raise NotImplementedError  # you do this
    #go through the tree using tree_policy
    with open("mcts_tree.pkl", "rb") as f:
        ogRoot = pickle.load(f)
    #print("Root:", ogRoot)
    root = traverse_tree(ogRoot, state)
    
    #ogRoot = root
    x = 0
    node = root
    if root is None:
         ogRoot = Node(state, [], np.where(state.legal)[0], 1, None, 1, 0)
         node = ogRoot
    while x < 1:
        if len(node.untriedActions) > 0:
            #print("Lenght:", len(node.untriedActions))
            newNode = tree_policy(node, cfg)
            if newNode is None:
                break
            delta = default_policy(newNode.state)
            backup(newNode, delta)
        else:
            if x == 0:
                #print("Testing")0
                root = node
            node = best_child(node, (1/np.sqrt(2)))
            #node = newNode
            #print("hello?")
            x+=1
    #if ogRoot == root:
        #print("is same")
    
    with open("mcts_tree.pkl", "wb") as f:
        pickle.dump(ogRoot, f)
    return best_child(root, (1/np.sqrt(2))).action

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
    
def tree_policy(node: Node, cfg) -> Node:
    while node.state.ended is False:
        if len(node.untriedActions) > 0:
            return expand (node)
        else:
            print("Test")
            return best_child(node, (1/np.sqrt(2)))
    #print ("what")
    #return tree_policy(node.parent, cfg)
        

        
    #Explore vs Exploit, something like 70/30 in favor of exploiting?


def expand(v: Node) -> Node:
    #add new node to tree
    #select action
    action = np.random.choice(v.untriedActions).item()
    #print("action:", action)
    assert action in v.untriedActions
    newState = env.step(v.state, action)
    v.untriedActions = np.delete(v.untriedActions, np.where(v.untriedActions==action))
    child = Node(newState, [], np.where(newState.legal)[0], action, v, 0, 0)
    #child.action = action
    #child.Parent = v
    v.children.append(child)
    return child
   


def best_child(root: Node, c) -> Node:
    bc = None
    val = 0
    for child in root.children:
        if child.UCT(c) > val:
            val = child.UCT(c)
            bc = child
    return bc
    #define (current) best child node of the tree


def default_policy(state: State) -> int:
    while state.ended is False:
        action = np.random.choice(np.where(state.legal)[0]).item()
        #print("action:", action)
        nState = env.step(state, action)
        state = nState
    return state.point
    #policy used by default to traverse the tree


def backup(node, delta) -> None:
    while node is not None:
        node.visitCount +=1
        node.state.point += delta
        delta = -delta
        node = node.parent
    #set the default policy of the previous node?


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
                
                if a is None:
                    a = monte_carlo(state, cfg)
                a = monte_carlo(state, cfg)
                #print("final action:", a)
                #print(state, end="\n\n")
                #values = [monte_carlo(env.step(state, a), cfg) for a in actions]
                #a = actions[np.argmax(values) if state.maxim else np.argmin(values)]

            case _:
                raise ValueError(f"Unknown player {state.player}")
        state = env.step(state, a)

    print(f"{['nobody', 'o', 'x'][state.point]} won", state, sep="\n")
