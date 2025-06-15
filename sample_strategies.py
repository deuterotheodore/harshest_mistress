import numpy as np

# Predefined strategies for agent initialization with abs(v)^T W v quadratic form
sample_strategies = {
    "AC": {  # Always Cooperate
        "cooperation_bias": np.float16(5.0),
        "linear_weights": np.array([0.5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float16),
        "remote_weights": np.zeros((2, 2), dtype=np.float16),
        "recent_weights": np.zeros((6, 6), dtype=np.float16)
    },
    "AD": {  # Always Defect
        "cooperation_bias": np.float16(-5.0),
        "linear_weights": np.array([0.5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float16),
        "remote_weights": np.zeros((2, 2), dtype=np.float16),
        "recent_weights": np.zeros((6, 6), dtype=np.float16)
    },
    "TFT": {  # Tit-for-Tat
        "cooperation_bias": np.float16(0.0),
        "linear_weights": np.array([0.5, 0, 0, 0, 0, 10, 0, 0], dtype=np.float16),
        "remote_weights": np.zeros((2, 2), dtype=np.float16),
        "recent_weights": np.zeros((6, 6), dtype=np.float16)
    },
    "TFTT": {  # Tit-for-Two-Tats
        "cooperation_bias": np.float16(5.0),
        "linear_weights": np.array([0.5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float16),
        "remote_weights": np.zeros((2, 2), dtype=np.float16),
        "recent_weights": np.zeros((6, 6), dtype=np.float16)
    },
    "GT": {  # Grim Trigger
        "cooperation_bias": np.float16(5.0),
        "linear_weights": np.array([-2, 0, 0, 0, 0, 0, 0, 0], dtype=np.float16),
        "remote_weights": np.zeros((2, 2), dtype=np.float16),
        "recent_weights": np.zeros((6, 6), dtype=np.float16)
    },
    "TFTT-Fair1": {  # Fair-minded TFTT, mild
        "cooperation_bias": np.float16(5.0),
        "linear_weights": np.array([0.5, 0, -1.0, -1.0, 0, 0, 0, 0], dtype=np.float16),
        "remote_weights": np.zeros((2, 2), dtype=np.float16),
        "recent_weights": np.zeros((6, 6), dtype=np.float16)
    },
    "TFTT-Fair2": {  # Fair-minded TFTT, moderate
        "cooperation_bias": np.float16(5.0),
        "linear_weights": np.array([0.5, 0, -2.0, -2.0, 0, 0, 0, 0], dtype=np.float16),
        "remote_weights": np.zeros((2, 2), dtype=np.float16),
        "recent_weights": np.zeros((6, 6), dtype=np.float16)
    },
    "TFTT-Fair3": {  # Fair-minded TFTT, strong
        "cooperation_bias": np.float16(5.0),
        "linear_weights": np.array([0.5, 0, -3.0, -3.0, 0, 0, 0, 0], dtype=np.float16),
        "remote_weights": np.zeros((2, 2), dtype=np.float16),
        "recent_weights": np.zeros((6, 6), dtype=np.float16)
    }
}

# Initialize weights
sample_strategies["TFT"]["recent_weights"][3, 3] = np.float16(2.5)  # opp_last1
sample_strategies["TFTT"]["recent_weights"][3, 4] = sample_strategies["TFTT"]["recent_weights"][4, 3] = np.float16(5)
sample_strategies["GT"]["remote_weights"][0, 0] = np.float16(-10)  # opp_action_avg
sample_strategies["TFTT-Fair1"]["recent_weights"][3, 4] = sample_strategies["TFTT-Fair1"]["recent_weights"][4, 3] = np.float16(5)
sample_strategies["TFTT-Fair2"]["recent_weights"][3, 4] = sample_strategies["TFTT-Fair2"]["recent_weights"][4, 3] = np.float16(5)
sample_strategies["TFTT-Fair3"]["recent_weights"][3, 4] = sample_strategies["TFTT-Fair3"]["recent_weights"][4, 3] = np.float16(5)


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

