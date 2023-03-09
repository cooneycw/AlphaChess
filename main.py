import chess
from config.config import Config, SimulationCounter, MoveCounter
from src_code.agent.agent import AlphaZeroChess, board_to_input


def main():
    # Initialize the config and agent
    config = Config()
    agent = AlphaZeroChess(config, redis_host='localhost', redis_port=6379)
    sim_counter = SimulationCounter()
    move_counter = MoveCounter()


    while True:
        # Get the current state of the board
        state = board_to_input(config, agent.board)

        # Get the best action to take
        action = agent.get_action(state)

        # Take the action and update the board state
        agent.board.push_uci(action)
        move_counter.increment()

        # Update the MCTS tree with the latest state and action
        agent.update_tree(state, action)


        # Update the network weights every 100 simulations or every 10 moves
        if simulations >= 100 or moves >= 10:
            states, policy_targets, value_targets = generate_training_data()
            agent.update_network(states, policy_targets, value_targets)
            simulations = 0
            moves = 0

if __name__ == '__main__':
    main()
