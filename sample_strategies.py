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
    "TFT": {  # Tit-for-Tat: Copy opponent's last action
        "cooperation_bias": np.float16(0.0),
        "linear_weights": np.array([0, 0, 0, 0, 0, 10.0, 0, 0], dtype=np.float16),
        "remote_weights": np.zeros((2, 2), dtype=np.float16),
        "recent_weights": np.zeros((6, 6), dtype=np.float16)
    },
    "TFTT": {  # Tit-for-Two-Tats: Defect after two consecutive defections
        "cooperation_bias": np.float16(5.0),
        "linear_weights": np.array([0, 0, 0, 0, 0, -2.0, -2.0, 0], dtype=np.float16),
        "remote_weights": np.zeros((2, 2), dtype=np.float16),
        "recent_weights": np.zeros((6, 6), dtype=np.float16)
    },
    "GT": {  # Grim Trigger: Defect permanently after opponent's first defection
        "cooperation_bias": np.float16(5.0),
        "linear_weights": np.array([-2.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float16),
        "remote_weights": np.array([[-10.0, 0], [0, 0]], dtype=np.float16),
        "recent_weights": np.zeros((6, 6), dtype=np.float16)
    },
    "TFTT-Fair1": {  # Fair-minded TFTT, mild
        "cooperation_bias": np.float16(5.0),
        "linear_weights": np.array([0.5, 0, -1.0, -1.0, 0, 0, 0, 0], dtype=np.float16),
        "remote_weights": np.zeros((2, 2), dtype=np.float16),
        "recent_weights": np.array([[0]*6 for _ in range(6)], dtype=np.float16)
    },
    "TFTT-Fair2": {  # Fair-minded TFTT, moderate
        "cooperation_bias": np.float16(5.0),
        "linear_weights": np.array([0.5, 0, -2.0, -2.0, 0, 0, 0, 0], dtype=np.float16),
        "remote_weights": np.zeros((2, 2), dtype=np.float16),
        "recent_weights": np.array([[0]*6 for _ in range(6)], dtype=np.float16)
    },
    "TFTT-Fair3": {  # Fair-minded TFTT, strong
        "cooperation_bias": np.float16(5.0),
        "linear_weights": np.array([0.5, 0, -3.0, -3.0, 0, 0, 0, 0], dtype=np.float16),
        "remote_weights": np.zeros((2, 2), dtype=np.float16),
        "recent_weights": np.array([[0]*6 for _ in range(6)], dtype=np.float16)
    },
    "Pavlov": {  # Win-Stay, Lose-Shift
        "cooperation_bias": np.float16(0.0),
        "linear_weights": np.array([0, 0, -5.0, 0, 0, 5.0, 0, 0], dtype=np.float16),
        "remote_weights": np.zeros((2, 2), dtype=np.float16),
        "recent_weights": np.zeros((6, 6), dtype=np.float16)
    },
    "FTFT": {  # Forgiving Tit-for-Tat
        "cooperation_bias": np.float16(2.5),
        "linear_weights": np.array([0, 0, 0, 0, 0, 5.0, 0, 0], dtype=np.float16),
        "remote_weights": np.zeros((2, 2), dtype=np.float16),
        "recent_weights": np.zeros((6, 6), dtype=np.float16)
    },
    "random": {  # Random 50/50
        "cooperation_bias": np.float16(0.0),
        "linear_weights": np.zeros(8, dtype=np.float16),
        "remote_weights": np.zeros((2, 2), dtype=np.float16),
        "recent_weights": np.zeros((6, 6), dtype=np.float16)
    },
    "STFT": {  # Suspicious Tit-for-Tat
        "cooperation_bias": np.float16(-5.0),
        "linear_weights": np.array([0, 0, 0, 0, 0, 10.0, 0, 0], dtype=np.float16),
        "remote_weights": np.zeros((2, 2), dtype=np.float16),
        "recent_weights": np.zeros((6, 6), dtype=np.float16)
    },
    "TFTTT": {  # Tit-for-Three-Tats
        "cooperation_bias": np.float16(5.0),
        "linear_weights": np.array([0, 0, 0, 0, 0, -2.0, -2.0, -2.0], dtype=np.float16),
        "remote_weights": np.zeros((2, 2), dtype=np.float16),
        "recent_weights": np.zeros((6, 6), dtype=np.float16)
    },
    "GTFT": {  # Generous Tit-for-Tat
        "cooperation_bias": np.float16(1.0),
        "linear_weights": np.array([0, 0, 0, 0, 0, 5.0, 0, 0], dtype=np.float16),
        "remote_weights": np.zeros((2, 2), dtype=np.float16),
        "recent_weights": np.zeros((6, 6), dtype=np.float16)
    },
#long-term "high-trust" related strategies
    "TC": { # Trusting-Cooperator: cooperate if good long-term experience
        "cooperation_bias": np.float16(0.0),
        "linear_weights": np.zeros(8, dtype=np.float16),
        "remote_weights": np.array([[5.0, 0], [0, 0]], dtype=np.float16),
        "recent_weights": np.zeros((6, 6), dtype=np.float16)
    },
    "MT": { #Mutual-Trust: cooperate if history of mutual cooperation
        "cooperation_bias": np.float16(0.0),
        "linear_weights": np.zeros(8, dtype=np.float16),
        "remote_weights": np.array([[2.5, 0], [0, 2.5]], dtype=np.float16),
        "recent_weights": np.zeros((6, 6), dtype=np.float16)
    },
    "TBV": {#Trust-but-Verify
        "cooperation_bias": np.float16(2.5),
        "linear_weights": np.array([0, 0, 0, 0, 0, -2.0, -2.0, 0], dtype=np.float16),
        "remote_weights": np.array([[2.5, 0], [0, 0]], dtype=np.float16),
        "recent_weights": np.zeros((6, 6), dtype=np.float16)
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
