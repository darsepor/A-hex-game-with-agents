
from game.game_logic import GameLogic
from game.player import SimpleAI, ANNAI

def run_matches(num_games=100):
    results = {"Player 1": 0, "Player 2": 0, "Draw": 0}
    for i in range(num_games):
        print(f"Game {i+1}")
        game_logic = GameLogic(size=5, players=[ANNAI, SimpleAI])
        while not game_logic.game_over and game_logic.steps < 500:
            game_logic.next_turn()
        #If player 1 has no cities, player 2 wins, else player 1 wins
        if len(game_logic.players[0].cities) == 0:
            results["Player 2"] += 1
        elif len(game_logic.players[1].cities) == 0:
            results["Player 1"] += 1
            
        else:
            results["Draw"] += 1
    print(f"Player 1 wins: {results['Player 1']}")
    print(f"Player 2 wins: {results['Player 2']}")
    print(f"Draws: {results['Draw']}")
if __name__ == "__main__":
    run_matches()