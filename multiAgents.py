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

# region Completed stuff

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


class AlphaBetaAgent(MultiAgentSearchAgent):
    def helper(self, game_state: connect4.GameState, depth, alpha, beta):
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
                new_score = self.helper(successor, depth - 1, alpha, beta)[1]
                # If maximizing, select.
                if new_score > val:
                    val = new_score
                    best_col = col
                # BETA cutoff
                if val >= beta:
                    break
                alpha = max(alpha, val)

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
                new_score = self.helper(successor, depth - 1, alpha, beta)[1]
                # If maximizing, select.
                if new_score < val:
                    val = new_score
                    best_col = col
                # ALPHA cutoff
                if val <= alpha:
                    break
                beta = min(beta, val)
            return best_col, val

    def getAction(self, gameState):
        res = self.helper(gameState, self.depth, -math.inf, math.inf)[0]
        return res
# endregion


class ExpectimaxAgent(MultiAgentSearchAgent):

    def helper(self, game_state, depth):
        # Rand col from optional moves.
        rnd_col = None
        if len(game_state.getLegalActions(gameUtil.AI)) > 0:
            optional_move_nodes = game_state.getLegalActions(gameUtil.AI)
            rnd_col = random.choice(optional_move_nodes)

        # If terminal, return it's evaluation
        if game_state.is_terminal() or depth == 0:
            return rnd_col, self.evaluationFunction(game_state)

        # AI's turn
        if game_state.turn == gameUtil.AI:
            # Min value for maximization
            val = -math.inf

            # Get valid actions
            optional_move_nodes = game_state.getLegalActions(gameUtil.AI)
            column = random.choice(optional_move_nodes)

            # Check each possible col
            for col in optional_move_nodes:
                # Create successor node for available node.
                successor = game_state.generateSuccessor(gameUtil.AI, col)
                successor.switch_turn(successor.turn)
                # Get score for node
                new_score = self.helper(successor, depth - 1)[1]
                # If maximizing, select.
                if new_score > val:
                    val = new_score
                    column = col
            return column, val

        # PLAYER's turn
        else:
            val = 0
            # Check each possible col
            for col in optional_move_nodes:
                # Create successor node for available node.
                successor = game_state.generateSuccessor(gameUtil.AI, col)
                successor.switch_turn(successor.turn)
                # Get score for node
                score = self.helper(successor, depth - 1)[1]
                # Calc probability to pick the successor,
                p = len(optional_move_nodes)
                # and sum it up.
                val += (1 / p) * score
            return rnd_col, val

    def getAction(self, gameState):
        return self.helper(gameState, self.depth)[0]