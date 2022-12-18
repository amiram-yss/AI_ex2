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
    def helper(self, gameState:connect4.GameState, depth):
        # Get rid of self.(...)
        depth = self.depth
        # Get optional moves.
        optional_moves = gameState.getLegalActions()
        # Is last move
        is_last_move = gameState.is_terminal()
        # As long as game runs:
        if not is_last_move and depth != 0:
            # If the game can end next turn
            if is_last_move:
                # Make the winning move, if possible.
                if gameState.winning(gameState.get_piece_player()):
                    return math.inf
                # Avoiding loosing next turn.
                if gameState.winning(gameState.get_opp_piece()):
                    return -math.inf
                # It's a tie.
                return 0
            else:
                return gameState.getScore()
        # If it's AI's turn -> maximization.
        if gameState.turn == gameUtil.AI:
            val = -math.inf
            column = random.choice(optional_moves)
            for col in optional_moves:
                row = gameState.get_next_open_row(col)
                next_move_state = gameState
                next_move_state.turn = gameUtil.PLAYER
                next_move_state.drop_piece(row, col, gameUtil.AI_PIECE)
                next_score = self.helper(next_move_state, depth - 1)
                if next_score > val:
                    val, column = next_score, col
            return column, val
        # If PLAYER's turn -> minimization.
        else:
            val = math.inf
            column = random.choice(optional_moves)
            for col in optional_moves:
                row = gameState.get_next_open_row(col)
                next_move_state = gameState
                next_move_state.turn = gameUtil.AI
                next_move_state.drop_piece(row, col, gameUtil.PLAYER_PIECE)
                next_score = self.helper(next_move_state, depth - 1)
                if next_score > val:
                    val, column = next_score, col
            return column, val

    def getAction(self, gameState):
        return self.helper(gameState, self.depth)


class AlphaBetaAgent(MultiAgentSearchAgent):
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
