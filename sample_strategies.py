import numpy as np

#current agent genome: all values are float16
#PD strategy:
# cooperation_bias (float): constant collaboration preference, dominates first move
# remote_weights (2x2 symmetric matrix): weights on own and opponent cooperation average over full game (long-term memory) 
# recent_weights (6x6 symmetric matrix): weights on own and opponent cooperation/defection over past 3 turns 
# contrition (float): biases weight of own past defection vs cooperation (high contrition means defections are weighed more strongly, depending on weight sign this may still influence behavior either way). Value 1.0 means symmetric weight on cooperation/defection
# bigotry (float): same as contrition, but relating to opponent's past defection/cooperation
#tribal dynamics
# identity (5-dim): vector modeling "ethnic identity", used to calculate Euclidean distance from opponent's identity
# ingroup_preference (float): weight affecting how much distance between identity biases behavior. Value 0 means opponent's identity doesn't influence decision.
#sexual dynamics
# sex (float, should be boolean): 0=male, 1=female, set randomly at agent generation (not mutated or inherited)
# male_bias (float): constant bias added to cooperation_bias, carried only by males and passed patrilineally, intended to model basic sexual dimorphism (e.g. more aggressive males)
# male_investment (float): male contribution to offspring energy (Brutpflege), capped at MALE_INVESTMENT_CAP=0.5 to model baseline biological cost of parturition to females
# mating_display (float): extra energy expended by males in order to increase chance of being chosen by females (bonus to energy as perceived by female choice algorithm)
# chivalry (float): constant cooperation bonus applied only by males interacting with females
# ladys_choice (4-dim): four weights for female mate choice, values [strength, fitness, caring, kinship]
#total 12 items undergoing mutation-selection: 8 float, 5-dim, 4-dim, 2x2, 6x6: total 57 float16, of which 16 redundant (symmetric matrices).

sample_strategies = {
    "AC": {
        "cooperation_bias": np.float16(5.0),
        "remote_weights": np.array([[0, 0], [0, 0]], dtype=np.float16),
        "recent_weights": np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ], dtype=np.float16),
        "contrition": np.float16(1.0),
        "bigotry": np.float16(1.0)
    },
    "AD": {
        "cooperation_bias": np.float16(-5.0),
        "remote_weights": np.array([[0, 0], [0, 0]], dtype=np.float16),
        "recent_weights": np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ], dtype=np.float16),
        "contrition": np.float16(1.0),
        "bigotry": np.float16(1.0)
    },
    "TFT": {
        "cooperation_bias": np.float16(5.0),
        "remote_weights": np.array([[0, 0], [0, 0]], dtype=np.float16),
        "recent_weights": np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 9, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ], dtype=np.float16),
        "contrition": np.float16(1.0),
        "bigotry": np.float16(1.0)
    },
    "TFTT": {
        "cooperation_bias": np.float16(4.0),
        "remote_weights": np.array([[0, 0], [0, 0]], dtype=np.float16),
        "recent_weights": np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.3, 9, 0],
            [0, 0, 0, 9, 0.2, 0],
            [0, 0, 0, 0, 0, 0]
        ], dtype=np.float16),
        "contrition": np.float16(1.0),
        "bigotry": np.float16(2.0)
    },
    "TFTTT": {
        "cooperation_bias": np.float16(4.0),
        "remote_weights": np.array([[0, 0], [0, 0]], dtype=np.float16),
        "recent_weights": np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 5.5, 5.5, 5.5],
            [0, 0, 0, 5.5, 5.5, 5.5],
            [0, 0, 0, 5.5, 5.5, 5.5]
        ], dtype=np.float16),
        "contrition": np.float16(1.0),
        "bigotry": np.float16(0.4)
    },
    "GT": {
        "cooperation_bias": np.float16(5.0),
        "remote_weights": np.array([[5, 0], [0, 0]], dtype=np.float16),
        "recent_weights": np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 5, 0, 0],
            [0, 0, 0, 0, 5, 0],
            [0, 0, 0, 0, 0, 5]
        ], dtype=np.float16),
        "contrition": np.float16(1.0),
        "bigotry": np.float16(2.0)
    },
    "TFTT-Fair": {
        "cooperation_bias": np.float16(5.0),
        "remote_weights": np.array([[0, 0], [0, 0]], dtype=np.float16),
        "recent_weights": np.array([
            [-2, 0, 0, 0, 0, 0],
            [0, -2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ], dtype=np.float16),
        "contrition": np.float16(0.5),
        "bigotry": np.float16(1.5)
    },
    "Pavlov": {
        "cooperation_bias": np.float16(4.5),
        "remote_weights": np.array([[0, 0], [0, 0]], dtype=np.float16),
        "recent_weights": np.array([
            [15, 0, 0, -10, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [-10, 0, 0, 6, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ], dtype=np.float16),
        "contrition": np.float16(1.0),
        "bigotry": np.float16(2.0)
    },
    "FTFT": {
        "cooperation_bias": np.float16(2.5),
        "remote_weights": np.array([[0, 0], [0, 0]], dtype=np.float16),
        "recent_weights": np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 5, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ], dtype=np.float16),
        "contrition": np.float16(1.0),
        "bigotry": np.float16(1.0)
    },
    "random": {
        "cooperation_bias": np.float16(0.0),
        "remote_weights": np.array([[0, 0], [0, 0]], dtype=np.float16),
        "recent_weights": np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ], dtype=np.float16),
        "contrition": np.float16(1.0),
        "bigotry": np.float16(1.0)
    },
    "GTFT": {
        "cooperation_bias": np.float16(2.5),
        "remote_weights": np.array([[0, 0], [0, 0]], dtype=np.float16),
        "recent_weights": np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 5, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ], dtype=np.float16),
        "contrition": np.float16(1.0),
        "bigotry": np.float16(0.5)
    },
    "TC": { #"trusting cooperator"
        "cooperation_bias": np.float16(0.0),
        "remote_weights": np.array([[5, 0], [0, 0]], dtype=np.float16),
        "recent_weights": np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ], dtype=np.float16),
        "contrition": np.float16(1.0),
        "bigotry": np.float16(1.0)
    },
    "MT": { #"mutual trust"
        "cooperation_bias": np.float16(0.0),
        "remote_weights": np.array([[2.5, 0], [0, 2.5]], dtype=np.float16),
        "recent_weights": np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ], dtype=np.float16),
        "contrition": np.float16(1.0),
        "bigotry": np.float16(1.0)
    },
    "TBV": { #"trust-but-verify"
        "cooperation_bias": np.float16(2.5),
        "remote_weights": np.array([[2.5, 0], [0, 0]], dtype=np.float16),
        "recent_weights": np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ], dtype=np.float16),
        "contrition": np.float16(1.0),
        "bigotry": np.float16(1.0)
    },
    "Claude": {
        "cooperation_bias": np.float16(1.0),
        "remote_weights": np.array([[0, 0], [0, 0]], dtype=np.float16),
        "recent_weights": np.array([
            [ 3.0,  0.0,  0.0, -3.0, -2.0,  0.0],
            [ 0.0,  2.0,  0.0, -2.0, -1.0,  0.0],
            [ 0.0,  0.0,  1.0, -1.0, -1.0,  0.0],
            [-3.0, -2.0, -1.0,  3.0,  3.0,  0.0],
            [-2.0, -1.0, -1.0,  3.0,  3.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.0,  0.0,  1.0]
        ], dtype=np.float16),
        "contrition": np.float16(1.0),
        "bigotry": np.float16(2.0)
    },
    "fair": {
        "cooperation_bias": np.float16(3.0),
        "remote_weights": np.array([[5, 2], [2, 10]], dtype=np.float16),
        "recent_weights": np.array([
            [ 4.0,  0.0,  0.0, -2.0, -2.0,  0.0],
            [ 0.0,  3.0,  0.0, -1.5, -1.5,  0.0],
            [ 0.0,  0.0,  0.0, -0.2, 0.0,  0.0],
            [-2.0, -1.5, -0.2,  2.5,  4.0,  0.0],
            [-2.0, -1.5, 0.0,  4.0,  1.5,  0.0],
            [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.5]
        ], dtype=np.float16),
        "contrition": np.float16(1.0),
        "bigotry": np.float16(1.2)
    },
    "evolved": {
        "cooperation_bias": np.float16(4.35),
        "remote_weights": np.array([[6.63, 2.47], [2.47, 10.28]], dtype=np.float16),
        "recent_weights": np.array([
            [ 4.58,  -0.64, -0.97,-1.96, -2.85, 0.84],
            [ -0.64,  3.12, -0.26, -1.18, -2.07,  0.24],
            [ -0.97, -0.26,  0.07, 0.51, -1.3, 0.21],
            [-1.96, -1.18, 0.51,  2.1,  3.3,  0.03],
            [-2.85, -2.07, -1.3,  3.3,  1.41, -0.28],
            [ 0.84,  0.24, -0.21, 0.03, -0.28,  0.07]
        ], dtype=np.float16),
        "contrition": np.float16(0.89),
        "bigotry": np.float16(1.34)
    }
}

#Predefined sexual strategies
#ladys_choice order:  [strength, fitness, caring, kinship]
sexual_strategies = {
    "peacock": {  # Extreme mating display, low investment
        "male_bias": np.float16(-1.0),
        "male_investment": np.float16(0.02),
        "mating_display": np.float16(20.0),
        "chivalry": np.float16(0.2),
        "ladys_choice": np.array([1.0, 0.1, 0.1, 0.1], dtype=np.float16)
    },
    "dutiful": {  # High investment, cooperative
        "male_bias": np.float16(1.0),
        "male_investment": np.float16(0.3),
        "mating_display": np.float16(5.0),
        "chivalry": np.float16(2.0),
        "ladys_choice": np.array([0.2, 0.2, 1.0, 0.2], dtype=np.float16)
    },
    "fascist": {  # blood+strength
        "male_bias": np.float16(-3.0),
        "male_investment": np.float16(0.1),
        "mating_display": np.float16(0.5),
        "chivalry": np.float16(4.0),
        "ladys_choice": np.array([1.0, 1.0, 0.0, 5.0], dtype=np.float16)
    },
    "balanced": {  # Moderate across all traits
        "male_bias": np.float16(0.0),
        "male_investment": np.float16(0.1),
        "mating_display": np.float16(5.0),
        "chivalry": np.float16(1.0),
        "ladys_choice": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float16)
    }
}
