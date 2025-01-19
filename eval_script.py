
from game.game_logic import GameLogic
from game.player import SimpleAI, CNNACAI

def run_matches(num_games=100):
    results = {"Player 1": 0, "Player 2": 0}
    for i in range(num_games):
        print(f"Game {i+1}")
        game_logic = GameLogic(size=4, players=[CNNACAI, CNNACAI])
        while not game_logic.game_over:
            game_logic.next_turn()
        #If player 1 has no cities, player 2 wins, else player 1 wins
        if len(game_logic.players[0].cities) == 0:
            results["Player 2"] += 1
            print("Simple AI wins!")
        else:
            results["Player 1"] += 1
            print("neural network wins!")
    print(f"Player 1 wins: {results['Player 1']}")
    print(f"Player 2 wins: {results['Player 2']}")

if __name__ == "__main__":
    run_matches()