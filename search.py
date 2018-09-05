# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


import heapq, random
class MyPriorityQueue(util.PriorityQueue):
    # field is the index of item(which is a tuple) that will be used to judge whether 2 item are equal
    def update(self, item, priority,field):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i[field] == item[field]:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

def __generalSearch(searchType,problem,heuristic):

    """
    Insert start into the frontier.
    Initialize the visited list to empty.
    while frontier is nonempty:
        current = remove node from frontier
        add current to the visited list
        If current node == end, return True.
        for every nbr of current node not in visited  
            insert nbr into frontier
    return False.

    REMOVAL from frontier:
    Last-in first-out: DFS First-in first-out: BFS

    INSERTION into frontier:
    Front of list: DFS End of list: BFS

    DUPLICATES in frontier:
    Replace with new: DFS Delete new: BFS
    """
    # Insert start into the frontier.
    if searchType=="dfs" :
        frontier = util.Stack()
    elif (searchType == "bfs") :
        frontier=util.Queue()
    elif (searchType == "ucs"  or  searchType=="aStar") :
        frontier=MyPriorityQueue()


    startPosition = problem.getStartState()
    # node used to store info about parent  (0  current's position, 1  parent's action,2 parent's index,3 parent's position,4 cost until this node,5 heuristic in this node )
    currentCost=0
    currentHeuristic=heuristic(startPosition,problem)
    newEval=currentCost+currentHeuristic
    currentNode = (startPosition, None, None, None,newEval,currentHeuristic)


    if(searchType=="bfs" or searchType =="dfs"):
        frontier.push(currentNode)
    elif (searchType=="ucs" or searchType=="aStar" ) :
        frontier.push(currentNode,newEval)

    frontierPositionList=[]
    if searchType == "dfs":
        frontierPositionList.append(startPosition)
    elif (searchType == "bfs" or searchType=="ucs" or searchType=="aStar"):
        frontierPositionList.insert(0,startPosition)

    # Initialize the visited list to empty.
    visitedList = []
    parentList = []

    # while frontier is nonempty:
    while frontier.isEmpty() == 0:
        # current = remove node from frontier
        currentNode = frontier.pop()
        if searchType == "dfs":
            frontierPositionList.pop()
        elif (searchType == "bfs" or searchType == "ucs" or searchType=="aStar"):
            frontierPositionList.pop()

        currentPosition = currentNode[0]
        currentEval = currentNode[4]
        currentHeuristic=currentNode[5]
        action = currentNode[1]
        parentPosition = currentNode[3]

        # add current to the visited list
        visitedList.append(currentPosition)
        parentList.append(currentNode)

        # If current node == end, return True.
        if problem.isGoalState(currentPosition) == 1:

            logging.debug("-" + str(currentPosition))

            actionList = []

            # TODO actionList empty?
            while action is not None:
                actionList.insert(0, action)
                logging.debug("-" + str(action) + "-" + str(parentPosition))
                currentNode = parentList[currentNode[2]]
                action = currentNode[1]
                parentPosition = currentNode[3]
            return actionList

        # for every nbr of current node not in visited
        successorList = problem.getSuccessors(currentPosition)
        for successor in successorList:
            successorPosition = successor[0]
            successorAction = successor[1]
            successorCost = successor[2]
            if successorPosition not in visitedList:
                currentCost=currentEval-currentHeuristic
                newCost=currentCost+successorCost
                newHeuristic=heuristic(successorPosition,problem)
                newEval=newCost+newHeuristic
                nbrNode = (successorPosition, successorAction, len(parentList) - 1, currentPosition, newEval, newHeuristic)

                if searchType == "dfs"  or ( searchType=="bfs" and successorPosition not in frontierPositionList)  :
                    # insert nbr into frontier
                    frontier.push(nbrNode)
                    if searchType == "dfs":
                        frontierPositionList.append(successorPosition)
                    elif (searchType == "bfs"):
                        frontierPositionList.insert(0, successorPosition)
                elif searchType=="ucs" or searchType=="aStar":
                    frontier.update(nbrNode,newEval,0)
                    frontierPositionList.insert(0, successorPosition)


    # return False.
    return None;


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    return __generalSearch("dfs",problem,nullHeuristic)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    return __generalSearch("bfs",problem,nullHeuristic)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    return __generalSearch("ucs", problem,nullHeuristic)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    return __generalSearch("aStar", problem,heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch


import sys
import logging
logging.basicConfig(level=logging.WARN,format='%(levelname)s [%(filename)s : %(funcName)s : %(lineno)d] %(message)s')

