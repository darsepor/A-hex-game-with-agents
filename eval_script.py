import sys
from game.game_logic import GameLogic
from game.player import SimpleAI, ANNAI


def run_evaluation_curriculum(num_games_per_size=20):
    player1_class = ANNAI
    player2_class = SimpleAI

    evaluation_map_sizes = [2, 3, 4, 5, 6, 7] # NOTE: we do not train on maps of radius 6 and more, 
                                           # however, it is interesting to see how the network generalizes on larger maps.

    

    overall_results_summary = {}

    for current_eval_size in evaluation_map_sizes:
        print(f"\n--- Evaluating on Map Size: {current_eval_size} ---")
        results_for_size = {player1_class.__name__: 0, player2_class.__name__: 0, "Draw": 0}
        
        for i in range(num_games_per_size):
            print(f"  Game {i+1}/{num_games_per_size} (Size {current_eval_size})")
            

            game_logic = GameLogic(size=current_eval_size, players=[player1_class, player2_class])
            
            while not game_logic.game_over and game_logic.steps < 500:
                game_logic.next_turn()
            
            winner_name = "Draw"
            if len(game_logic.players[0].cities) == 0: # Player 1 (ANNAI) lost
                winner_name = player2_class.__name__
            elif len(game_logic.players[1].cities) == 0: # Player 2 (SimpleAI) lost
                winner_name = player1_class.__name__
            
            results_for_size[winner_name] += 1

        print(f"\n  Results for Map Size {current_eval_size} ({num_games_per_size} games):")
        print(f"    {player1_class.__name__} (Player 1) wins: {results_for_size[player1_class.__name__]}")
        print(f"    {player2_class.__name__} (Player 2) wins: {results_for_size[player2_class.__name__]}")
        print(f"    Draws: {results_for_size['Draw']}")
        overall_results_summary[current_eval_size] = results_for_size

    print("\n--- Overall Evaluation Summary ---")
    for size, res in overall_results_summary.items():
        print(f"  Size {size}: {player1_class.__name__} Wins: {res[player1_class.__name__]}, {player2_class.__name__} Wins: {res[player2_class.__name__]}, Draws: {res['Draw']}")

if __name__ == "__main__":
    run_evaluation_curriculum()