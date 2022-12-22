"""
Introduction to Artificial Intelligence, 89570, Bar Ilan University, ISRAEL

Student name:   Amiram Yassif
Student ID:     314985474

"""

# multiAgents.py
# --------------
# Attribution Information: part of the code were created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# http://ai.berkeley.edu.
# We thank them for that! :)

# region useless stuff

import random, util, math

import numpy

import connect4
import gameUtil
from connect4 import Agent


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 1  # agent is always index 1
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class BestRandom(MultiAgentSearchAgent):

    def getAction(self, gameState):
        return gameState.pick_best_move()


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 1)
    """
    # TODO check how inheritance works in Python
    def helper(self, game_state: connect4.GameState, depth):
        # Is last move
        is_terminal = game_state.is_terminal()
        # As long as game runs:
        # winning = winning_move
        if is_terminal or depth == 0:
            return None, self.evaluationFunction(game_state)
        optional_move_nodes = game_state.getLegalActions()
        best_col = 0
        # AI
        if game_state.turn == gameUtil.AI:
            val = -math.inf
            # Check each possible col
            for col in optional_move_nodes:
                # Create successor node for available node.
                successor = game_state.generateSuccessor(game_state.turn, col)
                successor.switch_turn(game_state.turn)
                # Get score for node
                new_score = self.helper(successor, depth - 1)[1]
                # If maximizing, select.
                if new_score > val:
                    val = new_score
                    best_col = col
            return best_col, val
        # PLAYER
        else:
            val = math.inf
            # Check each possible col
            for col in optional_move_nodes:
                # Create successor node for available node.
                successor = game_state.generateSuccessor(game_state.turn, col)
                successor.switch_turn(game_state.turn)
                # Get score for node
                new_score = self.helper(successor, depth - 1)[1]
                # If maximizing, select.
                if new_score < val:
                    val = new_score
                    best_col = col
            return best_col, val

    def getAction(self, gameState):
        res = self.helper(gameState, self.depth)[0]
        return res

# endregion

class AlphaBetaAgent(MultiAgentSearchAgent):
    def helper(self, game_state: connect4.GameState, depth):
        # Is last move
        is_terminal = game_state.is_terminal()
        # As long as game runs:
        # winning = winning_move
        if is_terminal or depth == 0:
            return None, self.evaluationFunction(game_state)
        optional_move_nodes = game_state.getLegalActions()
        best_col = 0
        # AI
        if game_state.turn == gameUtil.AI:
            val = -math.inf
            # Check each possible col
            for col in optional_move_nodes:
                # Create successor node for available node.
                successor = game_state.generateSuccessor(game_state.turn, col)
                successor.switch_turn(game_state.turn)
                # Get score for node
                new_score = self.helper(successor, depth - 1)[1]
                # If maximizing, select.
                if new_score > val:
                    val = new_score
                    best_col = col
            return best_col, val
        # PLAYER
        else:
            val = math.inf
            # Check each possible col
            for col in optional_move_nodes:
                # Create successor node for available node.
                successor = game_state.generateSuccessor(game_state.turn, col)
                successor.switch_turn(game_state.turn)
                # Get score for node
                new_score = self.helper(successor, depth - 1)[1]
                # If maximizing, select.
                if new_score < val:
                    val = new_score
                    best_col = col
            return best_col, val

    def getAction(self, gameState):
        res = self.helper(gameState, self.depth)[0]
        return res

    def getAction(self, gameState):
        """
            Your minimax agent with alpha-beta pruning (question 2)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
