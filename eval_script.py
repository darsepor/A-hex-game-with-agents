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
    trained_sizes = [s for s in evaluation_map_sizes if s < 6]
    heldout_sizes = [s for s in evaluation_map_sizes if s >= 6]
    grouped_results = {
        'trained': {player1_class.__name__: 0, player2_class.__name__: 0, 'Draw': 0},
        'heldout': {player1_class.__name__: 0, player2_class.__name__: 0, 'Draw': 0}
    }
    for s in trained_sizes:
        for key in grouped_results['trained']:
            grouped_results['trained'][key] += overall_results_summary[s][key]
    for s in heldout_sizes:
        for key in grouped_results['heldout']:
            grouped_results['heldout'][key] += overall_results_summary[s][key]
    total_trained_games = num_games_per_size * len(trained_sizes)
    total_heldout_games = num_games_per_size * len(heldout_sizes)
    non_draw_trained_games = total_trained_games - grouped_results['trained']['Draw']
    non_draw_heldout_games = total_heldout_games - grouped_results['heldout']['Draw']
    print("\n--- Grouped Evaluation Summary ---")
    print(f"Trained map sizes {trained_sizes}:")
    print(f"  {player1_class.__name__} win rate (excluding draws): {grouped_results['trained'][player1_class.__name__] / non_draw_trained_games * 100:.2f}% ({grouped_results['trained'][player1_class.__name__]}/{non_draw_trained_games})")
    print(f"  {player2_class.__name__} win rate (excluding draws): {grouped_results['trained'][player2_class.__name__] / non_draw_trained_games * 100:.2f}% ({grouped_results['trained'][player2_class.__name__]}/{non_draw_trained_games})")
    print(f"  Draw rate: {grouped_results['trained']['Draw'] / total_trained_games * 100:.2f}% ({grouped_results['trained']['Draw']}/{total_trained_games})")
    print(f"Held-out map sizes {heldout_sizes}:")
    print(f"  {player1_class.__name__} win rate (excluding draws): {grouped_results['heldout'][player1_class.__name__] / non_draw_heldout_games * 100:.2f}% ({grouped_results['heldout'][player1_class.__name__]}/{non_draw_heldout_games})")
    print(f"  {player2_class.__name__} win rate (excluding draws): {grouped_results['heldout'][player2_class.__name__] / non_draw_heldout_games * 100:.2f}% ({grouped_results['heldout'][player2_class.__name__]}/{non_draw_heldout_games})")
    print(f"  Draw rate: {grouped_results['heldout']['Draw'] / total_heldout_games * 100:.2f}% ({grouped_results['heldout']['Draw']}/{total_heldout_games})")

if __name__ == "__main__":
    run_evaluation_curriculum()