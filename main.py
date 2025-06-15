import numpy as np
import random
from collections import defaultdict, deque
from multiprocessing import Process, Queue
import psutil #just to ask for number of physical cores
import pygame
import asyncio
import platform
import time

from sample_strategies import sample_strategies, sexual_strategies

#todo / ideas 
#
#game rules:
# flexible board size (e.g. 5x5 to 20x20)
# climate: areas with higher/lower energy, e.g. energy desert at the center or in a corner
# reproduction/migration: always spawn offspring in own cell, encode migration bias or threshold in genome to make migration a "choice"
#
#pygame graphical interface:
# scatter plot e.g. showing cooperation ratio vs energy gained
# input mask/sliders to enter parameter values directly, start/stop button, button to save agent log to file 

NUM_CORES = psutil.cpu_count(logical=False)
if NUM_CORES is None:
    NUM_CORES = 8 # observed about 4.5x speedup for 8 cores

#the game
NUM_TURNS = 400
INTERACTION_DEPTH = 3 #interactions per turn

# payoff matrix: standard Prisoner's Dilemma scores
# 1 for cooperate, -1 for defect, value 0 (no history), make sure to never look up other values (0 = "no history")
PAYOFFS = {(1, 1): 3, (1, -1): 0, (-1, 1): 5, (-1, -1): 1}

#mutation rates
MUTATION_RATE = 0.5
MUTATION_RATE_IDENTITY = 0.1
MUTATION_RATE_SEXUAL = 0.1
#migration
MIGRATION_RATE = 0.05
MIGRATION_COST = 5
#energy ecology
CONSUMPTION = 25
AGING_PENALTY = 0.4 #set to 0 for immortality
EXTRACTION_BASE = 100 # base value of energy harvested in each cell per turn, modified by bonus/penalty system below

#prisoner's dilemma setup
#reward/penalty system defined here is relevant for evolutionary pressure, the PAYOFFS matrix is only for keeping track of nominal scores without any direct environmental impact (but used for "fitness" in female choice in mating)
COOPERATION_BONUS = 6/5 # agents are rewarded for cooperating by increasing energy harvest, should be 6/5 based on standard PAYOFFS values
DEFECTION_PENALTY = 2/5 # agents are penalized for defecting by reducing energy harvest, should be 2/5 based on standard PAYOFFS values
IDENTITY_DIMENSIONS = 5 #must be >=2. choice of 5 is just an intuition at this point, examine dynamics to optimize

#reproduction parameters
ALLOW_ASEXUAL = False
ASEXUAL_REPRODUCTION_THRESHOLD = 100
ASEXUAL_REPRODUCTION_COST = 10
ALLOW_SEXUAL = True
SEXUAL_REPRODUCTION_THRESHOLD = 100
SEXUAL_REPRODUCTION_COST = 5
OFFSPRING_ENERGY = 50
MALE_INVESTMENT_CAP = 0.5 #max male investment in reproduction cost (Brutpflege)
MATING_DISPLAY_EFFECTIVENESS = 2.0 #effect of male investment in mating display
BULLYING_EFFECTIVENESS = 2.0 #brute-force impact of male strength on reproductive chance, overriding female selection algorithm


#display, console logging
DRAW_INTERVAL = 2 #only update graphics every nth turn
FPS = 10 #max fps for graphical output. simulation will slow down if this value is reached
PRINT_INTERVAL = 20 #print out turn statistics every nth turn
DEBUG_TURNS = 2 #extra detailed printout for the first few turns
LOG_TO_FILE = True
LOG_INTERVAL = 50 # if LOG_TO_FILE, append to log every nth turn
logfile_path = "/tmp/dead_agents_log.txt"

# Pygame visualization setup
WIDTH, HEIGHT = 1920, 1080
GRID_SIZE = 10
CELL_SIZE = 80
GRID_OFFSET_X, GRID_OFFSET_Y = 40, 50
GRAPH_OFFSET_X, GRAPH_OFFSET_Y = 880, 50
GRAPH2_OFFSET_X, GRAPH2_OFFSET_Y = 1400, 50
HIST_OFFSET_X, HIST_OFFSET_Y = 880, 400
HIST2_OFFSET_X, HIST2_OFFSET_Y = 1400, 400
DATA_OFFSET_X, DATA_OFFSET_Y = 900, 750
BUTTON_OFFSET_X, BUTTON_OFFSET_Y = 900, 980
WHITE = (250, 250, 250)
BLACK = (5, 5, 5)
GREEN = (20, 255, 30)
RED = (255, 20, 30)
BLUE = (50, 50, 255)
GRAY = (150, 150, 150)
YELLOW = (250, 250, 10)

#the total_score and games_played arrays used to be part of the Agent structure and had to be removed for parallelization, because they are updated mid-turn for agents with more than one interaction
total_score = defaultdict(float)
games_played =  defaultdict(int)

class Agent:
    def __init__(self, posx=None, posy=None, energy=30, id=None, parent=None, born=None, cooperation_bias=None, linear_weights=None, 
                 remote_weights=None, recent_weights=None, identity=None, ingroup_preference=None, sex=None, male_bias=None, male_investment=None, mating_display=None, chivalry=None, ladys_choice=None):
        self.posx = posx
        self.posy = posy
        self.energy = energy
        self.id = id
        self.parent = parent if parent is not None else []  # List: [], [clone], or [father, mother]
        self.offspring = []
        self.born = born
        self.sex = random.randint(0, 1) if sex is None else sex  # 0=male, 1=female
#        self.games_played = 0 #removed for parallelization
#        self.total_score = 0 #removed for parallelization
        if cooperation_bias is None:
            self.cooperation_bias = np.float16(np.random.uniform(-5, 5))
        else:
            self.cooperation_bias = np.float16(cooperation_bias)
        if linear_weights is None:
            self.linear_weights = np.random.uniform(-2, 2, 8).astype(np.float16)
        else:
            self.linear_weights = np.array(linear_weights, dtype=np.float16)
        if remote_weights is None:
            self.remote_weights = np.zeros((2, 2), dtype=np.float16)
            self.remote_weights[0, 0] = np.random.uniform(-1, 1)
            self.remote_weights[1, 1] = np.random.uniform(-1, 1)
            self.remote_weights[0, 1] = self.remote_weights[1, 0] = np.random.uniform(-1, 1)
        else:
            self.remote_weights = np.array(remote_weights, dtype=np.float16)
        if recent_weights is None:
            self.recent_weights = np.zeros((6, 6), dtype=np.float16)
            for i in range(6):
                self.recent_weights[i, i] = np.random.uniform(-0.5, 0.5)
                for j in range(i + 1, 6):
                    self.recent_weights[i, j] = self.recent_weights[j, i] = np.random.uniform(-0.5, 0.5)
        else:
            self.recent_weights = np.array(recent_weights, dtype=np.float16)
        if identity is None:
            self.identity = np.random.uniform(0, 10, 5).astype(np.float16)
        else:
            self.identity = np.array(identity, dtype=np.float16)
        if ingroup_preference is None:
            self.ingroup_preference = np.random.uniform(0, 1)
        else:
            self.ingroup_preference = np.array(ingroup_preference, dtype=np.float16)
        self.male_bias = np.float16(random.uniform(-5, 5)) if male_bias is None else np.float16(male_bias) if self.sex == 0 else np.float16(0.0)  # 0 for females
        if male_investment is None:
            self.male_investment = np.float16(np.random.uniform(0, MALE_INVESTMENT_CAP))  # Percentage
        else:
            self.male_investment = np.float16(male_investment)
        if mating_display is None:
            self.mating_display = np.float16(np.random.uniform(0, 10))  # Energy
        else:
            self.mating_display = np.float16(mating_display)
        if chivalry is None:
            self.chivalry = np.float16(np.random.uniform(0, 5))  # Interaction total_output
        else:
            self.chivalry = np.float16(chivalry)
        if ladys_choice is None:
            self.ladys_choice = np.random.uniform(0, 1, 4).astype(np.float16)  # Weights vector, order [strength, fitness, caring, kinship]
        else:
            self.ladys_choice = np.array(ladys_choice, dtype=np.float16)

# order of values in input_vector:
# [opp_remote, own_remote, self_last1, self_last2, self_last3, opp_last1, opp_last2, opp_last3]
#we need this to model the theoretical expression of known strategies such as AC, AD, TFT, TFTT, GT
#all of these completely disregard own history, and AC, AD ignore history completely, and TFT only relies on opp_last1

    def decide(self, input_vector, distance, opp_sex):
        # input_vector: [opp_action_avg, own_action_avg, self_last1/2/3, opp_last1/2/3]
        # action_avg: [-1, 1], -1 = always defected, 1 = always cooperateed
        # self_last1/2/3, opp_last1/2/3: 1 (cooperated), -1 (defected), 0 (no history)
        #note: weights should naturally be in the range [-5,5] since we interpret total_result to represent probabilistic action in the [-5,5] range, deterministic outside of this. Weights can in principle "co-evolve" to larger values that may cancel out and still give probabilistic results, so pending extensive tests of evolutionary dynamics, let's not impose hard caps, but it will probably make sense to eventually constrain them to some range, such as [-10,10].

        if input_vector[2] == 0:	#empty history 
            total_output=self.cooperation_bias
        else:
	    #apply strategy            
            linear_term = np.dot(input_vector, self.linear_weights)
            remote_vector = input_vector[:2]  # [-1, 1]
            recent_vector = input_vector[2:]  # {-1, 1, 0}
            # Apply strategy
            linear_term = np.dot(input_vector, self.linear_weights)
            # we are now using an "anti-quadratic form", as in abs(v)^T W v
            remote_term = np.dot(np.abs(remote_vector), np.dot(self.remote_weights, remote_vector))
            recent_term = np.dot(np.abs(recent_vector), np.dot(self.recent_weights, recent_vector))
            total_output = linear_term + remote_term + recent_term + self.cooperation_bias

        # sexual dimorphism mechanic
        if ALLOW_SEXUAL:
            if self.sex == 0:  
                total_output += self.male_bias
                if opp_sex == 1:
                    total_output += self.chivalry

        # xenophobia mechanic
        xenophobia = distance * self.ingroup_preference #small distance and small ingroup_preference reduces xenophobia
        total_output *= np.clip(1 - 2 * (xenophobia - 0.5), 0, 1) # value reduced to 0 if xenophobia=1, doubled if 0, left alone if 0.5

        #Values <-5 map to 0, values >+5 to 1. This is historically due to the sigmoid (logistic) function we used earlier, yielding values of 0.007 and .993 for -5, 5 respectively, which we treated as the "boundary to determinism". We could in principle divide everything by five for an equivalent algorithm, but there is also nothing wrong with keeping the [-5,5] convention.
        prob_cooperate = np.clip((0.5 + 0.1 * total_output), 0, 1)
        return 1 if random.random() < prob_cooperate else -1

def is_interacting(pos1x, pos1y, pos2x, pos2y):
    return max(abs(pos1x - pos2x), abs(pos1y - pos2y)) <= 1

def mutate_genome(cooperation_bias, linear_weights, remote_weights, recent_weights, identity, ingroup_preference, male_bias, male_investment, mating_display, chivalry, ladys_choice, parent2_genome=None):

    # If parent2_genome provided (sexual reproduction), average genomes
    if parent2_genome:
        cooperation_bias = (cooperation_bias + parent2_genome['cooperation_bias']) / 2
        linear_weights = (linear_weights + parent2_genome['linear_weights']) / 2
        remote_weights = (remote_weights + parent2_genome['remote_weights']) / 2
        recent_weights = (recent_weights + parent2_genome['recent_weights']) / 2
        identity = (identity + parent2_genome['identity']) / 2
        ingroup_preference = (ingroup_preference + parent2_genome['ingroup_preference']) / 2
 
    mutated_cooperation_bias = (cooperation_bias + np.random.normal(0, MUTATION_RATE)).astype(np.float16)
    mutated_linear = linear_weights + np.random.normal(0, MUTATION_RATE, size=8).astype(np.float16)
    mutated_remote = remote_weights.copy()
    mutated_remote[0, 0] += np.random.normal(0, MUTATION_RATE)
    mutated_remote[1, 1] += np.random.normal(0, MUTATION_RATE)
    delta = np.random.normal(0, MUTATION_RATE)
    mutated_remote[0, 1] = mutated_remote[1, 0] = mutated_remote[0, 1] + delta
    mutated_recent = recent_weights.copy()
    for i in range(6):
        mutated_recent[i, i] += np.random.normal(0, MUTATION_RATE)
        for j in range(i + 1, 6):
            delta = np.random.normal(0, MUTATION_RATE)
            mutated_recent[i, j] = mutated_recent[j, i] = mutated_recent[i, j] + delta
    mutated_identity = np.clip(identity + np.random.normal(0, MUTATION_RATE_IDENTITY, size=5), 0, 10).astype(np.float16)
    mutated_ingroup_preference = np.clip(ingroup_preference + np.random.normal(0, MUTATION_RATE_IDENTITY), 0, 10).astype(np.float16)
    if ALLOW_SEXUAL:
        mutated_male_bias = np.float16(male_bias + np.random.normal(0, MUTATION_RATE_SEXUAL))
        mutated_male_investment = np.clip(male_investment + np.random.normal(0, MUTATION_RATE_SEXUAL), 0, MALE_INVESTMENT_CAP).astype(np.float16)
        mutated_mating_display = np.clip(mating_display + np.random.normal(0, MUTATION_RATE_SEXUAL), 0, SEXUAL_REPRODUCTION_THRESHOLD).astype(np.float16)
        mutated_chivalry = np.float16(chivalry + np.random.normal(0, MUTATION_RATE_SEXUAL))
        mutated_ladys_choice = ladys_choice + np.random.normal(0, MUTATION_RATE_SEXUAL, size=4).astype(np.float16)

    return mutated_cooperation_bias, mutated_linear, mutated_remote, mutated_recent, mutated_identity, mutated_ingroup_preference, mutated_male_bias

def agent_snapshot(agent, games_played, total_score):
    return {
        'id': agent.id,
        'born': agent.born,
        'died': turn,
        'posx': agent.posx,
        'posy': agent.posy,
        'parent': agent.parent,
        'sex': agent.sex,
        'offspring': list(agent.offspring),
        'games_played': games_played[agent.id], #recover from external array
        'total_score': total_score[agent.id], #recover from external array
        'cooperation_bias': agent.cooperation_bias.tolist(),
        'linear_weights': agent.linear_weights.tolist(),
        'remote_weights': agent.remote_weights.tolist(),
        'recent_weights': agent.recent_weights.tolist(),
        'identity': agent.identity.tolist(),
        'ingroup_preference': float(agent.ingroup_preference),
        'male_bias': agent.male_bias,
        'male_investment': agent.male_investment,
        'mating_display': agent.mating_display,
        'chivalry': agent.chivalry,
        'ladys_choice': agent.ladys_choice
    }
    

def append_agent(posx, posy, id, born, identity, ingroup_preference, strategy, sexual_selection, agents, sex=None):
    """Appends an Agent to the agents list with interaction and sexual selection strategy parameters."""
    if strategy not in sample_strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(sample_strategies.keys())}")
    if sexual_selection not in sexual_strategies:
        raise ValueError(f"Unknown sexual selection: {sexual_selection}. Available: {list(sexual_strategies.keys())}")
    
    strategy_params = sample_strategies[strategy]
    sexual_params = sexual_strategies[sexual_selection]
    agent_sex = sex if sex is not None else random.randint(0, 1)
    agent = Agent(
        posx=posx,
        posy=posy,
        id=id,
        born=born,
        cooperation_bias=strategy_params["cooperation_bias"],
        linear_weights=strategy_params["linear_weights"].copy(),
        remote_weights=strategy_params["remote_weights"].copy(),
        recent_weights=strategy_params["recent_weights"].copy(),
        identity=np.array(identity, dtype=np.float16),
        ingroup_preference=np.float16(ingroup_preference),
        sex=agent_sex,
        male_bias=np.float16(sexual_params["male_bias"]) if agent_sex == 0 else np.float16(0.0),
        male_investment=sexual_params["male_investment"],
        mating_display=sexual_params["mating_display"],
        chivalry=sexual_params["chivalry"],
        ladys_choice=sexual_params["ladys_choice"].copy()
    )
    agents.append(agent)
    return agent
        
def process_chunk(pairs, result_queue, history_dict):
    """process_chunk is so named because it is passed the n=NUM_CORES 'chunks' of the interacting_pairs array for parallelization. 
    The name is deceivingly prosaic, as this contains the main prisoner's dilemma mechanics. Agent's "decide" functions are called, cooperation bonus and defection penalty are applied, also if sexual selection is enabled, possible mating pairs are picked, but offspring is generated later because this can't be done in parallel (or the same mother would generate an inane number of offspring because she is listed in mating pairs distributed among the 'chunks')."""
    local_payouts = []
    local_history_updates = {}
    local_interactions = defaultdict(int)
    local_total_score = defaultdict(float)
    local_games_played = defaultdict(int)
    local_cc_games = defaultdict(int)
    local_dd_games = defaultdict(int)    
    local_mating_pairs = []
        
    for A, B in pairs:
        key = (min(A.id, B.id), max(A.id, B.id))
        
        # Get history data: (deque, A_action_avg, B_action_avg, A_action_sum, B_action_sum, total_interactions)
        history, A_action_avg, B_action_avg, total_interactions = history_dict.get(
            key, (deque(maxlen=3), 0.0, 0.0, 0)
        )
        history = history.copy()
        
        distance =  np.sum((A.identity - B.identity)**2)/ IDENTITY_DIMENSIONS #technically, euclideaan distance squared, try quadratic penalty on increasing "identity distance"
        score_A=0
        score_B=0
        modifier=0

        for i in range (INTERACTION_DEPTH):            
            # Prepare history data for decide function
            A_last1 = history[-1][0] if len(history) >= 1 else 0
            A_last2 = history[-2][0] if len(history) >= 2 else 0
            A_last3 = history[-3][0] if len(history) >= 3 else 0
            B_last1 = history[-1][1] if len(history) >= 1 else 0
            B_last2 = history[-2][1] if len(history) >= 2 else 0
            B_last3 = history[-3][1] if len(history) >= 3 else 0
            
            input_vector_A = np.array(
                [B_action_avg, A_action_avg, A_last1, A_last2, A_last3, B_last1, B_last2, B_last3],
                dtype=np.float16
            )
            input_vector_B = np.array(
                [A_action_avg, B_action_avg, B_last1, B_last2, B_last3, A_last1, A_last2, A_last3],
                dtype=np.float16
            )

            action_A = A.decide(input_vector_A, distance, B.sex)  # Returns 1 or -1
            action_B = B.decide(input_vector_B, distance, A.sex)  # Returns 1 or -1 
 
            score_A += PAYOFFS[(action_A, action_B)]
            score_B += PAYOFFS[(action_B, action_A)]
            #"score" is only recorded for statistics, it is irrelevant for evolutionary dynamics. Actual bonus/penalty for cooperation/defection is issued here:
            #addition for each interaction in INTERACTION_DEPTH, divided by INTERACTION_DEPTH to get average after the loop concludes
            if action_A == 1 and action_B == 1:  # CC
                modifier += COOPERATION_BONUS  # standard 6/5=1.2
            elif action_A == -1 and action_B == -1:  # DD
                modifier += DEFECTION_PENALTY  # standard 2/5=0.4
            else:  # CD or DC
                modifier += 1.0 #base value (no bonus/penalty)
    
            history.append((action_A, action_B))

            local_interactions[(A.posx, A.posy)] += 1
            local_interactions[(B.posx, B.posy)] += 1
            local_games_played[A.id] += 1
            local_games_played[B.id] += 1
            if action_A == 1 and action_B == 1:  # CC
                local_cc_games[(A.posx, A.posy)] += 1
                local_cc_games[(B.posx, B.posy)] += 1
            elif action_A == -1 and action_B == -1:  # DD
                local_dd_games[(A.posx, A.posy)] += 1
                local_dd_games[(B.posx, B.posy)] += 1

            # Update past actions averages for remote history
            total_interactions += 1
            A_action_avg = np.float16((A_action_avg * (total_interactions - 1) + action_A) / total_interactions)
            B_action_avg = np.float16((B_action_avg * (total_interactions - 1) + action_B) / total_interactions)
 
        local_total_score[A.id] += score_A
        local_total_score[B.id] += score_B        
        local_payouts.append((A.id, B.id, score_A, score_B, modifier/INTERACTION_DEPTH))
        local_history_updates[key] = (history, A_action_avg, B_action_avg, total_interactions)
  
        # Sexual reproduction: collect possible mating pairs

        if ALLOW_SEXUAL and ((A.sex == 0 and B.sex == 1) or (A.sex == 1 and B.sex == 0)):
            male, female = (A, B) if A.sex == 0 else (B, A)
            if (female.energy > SEXUAL_REPRODUCTION_THRESHOLD  and male.energy > (OFFSPRING_ENERGY + SEXUAL_REPRODUCTION_COST) * male.male_investment):
                local_mating_pairs.append((male.id, female.id))

    result_queue.put((
        local_payouts,
        local_history_updates,
        local_interactions,
        local_total_score,
        local_games_played,
        local_cc_games,
        local_dd_games,
        local_mating_pairs
    ))

def run_interactions(interacting_pairs, history_dict, interaction_payouts, interactions, total_score, games_played, cc_games, dd_games, next_id):
    """Parallelization for pairwise interaction, interacting_pairs is split into 'chunks', code for processing interactions is in process_chunks.
    process_chunk also marks pairs for possible sexual reproduction, but actual mating behavior in candidate pairs is implemented here.  """
    num_processes = min(NUM_CORES, len(interacting_pairs))  # Avoid empty chunks in case of extremely low population 

    # Split pairs into chunks
    pair_array = np.array(interacting_pairs, dtype=object)
    chunks = np.array_split(pair_array, num_processes)
    
    # Create a queue for results
    result_queue = Queue()
    processes = []
    mating_pairs = []
    
    # Start one process per chunk
    for chunk in chunks:
        p = Process(target=process_chunk, args=(chunk, result_queue, history_dict))
        processes.append(p)
        p.start()
    
    # Collect results
    for _ in range(num_processes):
        local_payouts, local_history_updates, local_interactions, local_total_score, local_games_played, local_cc_games, local_dd_games, local_mating_pairs = result_queue.get()

        interaction_payouts.extend(local_payouts)
        mating_pairs.extend(local_mating_pairs)
        for key, (history, A_action_avg, B_action_avg, total_interactions) in local_history_updates.items():
            history_dict[key] = (history, A_action_avg, B_action_avg, total_interactions)            
        for pos, count in local_interactions.items():
            interactions[pos] += count
        for pos, count in local_cc_games.items():
            cc_games[pos] += count
        for pos, count in local_dd_games.items():
            dd_games[pos] += count
        for agent_id, score in local_total_score.items():
            total_score[agent_id] += score
        for agent_id, num in local_games_played.items():
            games_played[agent_id] += num

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Deal with potential mating pairs returned in mating_pairs
    #sort by female id, allow only one child per mother per turn
    suitors = defaultdict(list)
    for pair in mating_pairs:
        suitors[pair[1]].append(pair[0])  # pair[1]: female_id, pair[0]: male_id
    if (turn % PRINT_INTERVAL == 0 or turn < DEBUG_TURNS):
        print (f"turn {turn}: {len(mating_pairs)} potential pairings, involving {len(suitors)} fertile females")    
    # Create one offspring per female
    for mother_id, male_ids in suitors.items():
        if mother_id not in id_to_agent:
            continue
        mother = id_to_agent[mother_id]
        # Triage: Limit to top 5 males by energy (experimental)
#        male_ids = sorted(male_ids, key=lambda mid: id_to_agent[mid].energy, reverse=True)[:5]
        # Probabilistic mate selection using ladys_choice
        scores = []
        for male_id in male_ids:
            male = id_to_agent.get(male_id)
            if not male or male.energy < (OFFSPRING_ENERGY + SEXUAL_REPRODUCTION_COST) * male.male_investment + male.mating_display:
                continue
            # Strength: energy modified by mating display investment
            strength = (male.energy + male.mating_display * MATING_DISPLAY_EFFECTIVENESS) / 500  # Normalize (~100–500)
            age = max(turn - male.born, 1)  # Avoid division by zero
            #Fitness: total score tempered by age, modified by number of offspring
            fitness = (total_score[male.id] / np.sqrt(age) * (1 + 0.1 * len(male.offspring))) / 100  # Normalize (~0–100)
            # Caring: Sum of interaction history
            key = (min(male_id, mother_id), max(male_id, mother_id))
            hist_data = history_dict.get(key, (deque(maxlen=3), 0, 0, 0))
            opp_avg = hist_data[1 if male_id < mother_id else 2]
            opp_last = [act[0 if male_id < mother_id else 1] for act in hist_data[0]] + [0] * (3 - len(hist_data[0]))
            caring = (opp_avg + sum(opp_last)) / 5  # Normalize (~-5 to 5)
            # Kinship: Quadratic identity distance
            kinship = np.sum((male.identity - mother.identity) ** 2) / IDENTITY_DIMENSIONS / 100  # Normalize (~0–500)
            # Score: Scalar product with ladys_choice
            score = np.dot(mother.ladys_choice, [strength, fitness, caring, kinship])
            scores.append((male_id, score, male.energy))
        if not scores:
            continue
    
        # Apply male competitiveness scaling
        male_ids = [s[0] for s in scores]
        scaled_scores = np.array([s[1] * (SEXUAL_REPRODUCTION_THRESHOLD + s[2] * BULLYING_EFFECTIVENESS) for s in scores])  # s[2]: male.energy
        probs = np.exp(scaled_scores - scaled_scores.max()) / np.sum(np.exp(scaled_scores - scaled_scores.max()))
        father_id = np.random.choice(male_ids, p=probs)
        father = id_to_agent[father_id]
    
        # Deduct energy
        mother.energy -= (OFFSPRING_ENERGY + SEXUAL_REPRODUCTION_COST) * (1 - father.male_investment)
        # Deduct mating_display penalty only for successful procreation, it might be more realisic ecologically to impose it on all candidates
        father.energy -= (OFFSPRING_ENERGY + SEXUAL_REPRODUCTION_COST) * father.male_investment + father.mating_display
  
        # Generate offspring genome
        offspring_sex = random.randint(0, 1)
        mutated_cooperation_bias, mutated_linear, mutated_remote, mutated_recent, \
        mutated_identity, mutated_ingroup_preference, mutated_male_bias = mutate_genome(
            father.cooperation_bias, father.linear_weights, father.remote_weights, father.recent_weights,
            father.identity, father.ingroup_preference, father.male_bias, father.male_investment, father.mating_display, father.chivalry, father.ladys_choice,
            parent2_genome={
                'cooperation_bias': mother.cooperation_bias,
                'linear_weights': mother.linear_weights,
                'remote_weights': mother.remote_weights,
                'recent_weights': mother.recent_weights,
                'identity': mother.identity,
                'ingroup_preference': mother.ingroup_preference,
                'male_investment': mother.male_investment,
                'mating_display': mother.mating_display,
                'chivalry': mother.chivalry,
                'ladys_choice': mother.ladys_choice
            }
        )
        offspring_male_bias = mutated_male_bias if offspring_sex == 0 else 0.0
        
        # Deduct energy 
        mother.energy -= (OFFSPRING_ENERGY + SEXUAL_REPRODUCTION_COST) * (1 - father.male_investment)
        father.energy -= (OFFSPRING_ENERGY + SEXUAL_REPRODUCTION_COST) * father.male_investment
        father.energy -= father.mating_display #deduct only at successful reproduction, or already in "suitor" phase?
        # Create offspring
        new_agent = Agent(
            posx=mother.posx, posy=mother.posy, energy=OFFSPRING_ENERGY,
            id=next_id, parent=(father_id, mother_id), born=turn,
            cooperation_bias=mutated_cooperation_bias,
            linear_weights=mutated_linear,
            remote_weights=mutated_remote,
            recent_weights=mutated_recent,
            identity=mutated_identity,
            ingroup_preference=mutated_ingroup_preference,
            sex=offspring_sex,
            male_bias=offspring_male_bias
        )
        agents.append(new_agent)
        id_to_agent[next_id] = new_agent
        #update 'offspring' list in both parents 
        for parent_id in (father_id, mother_id):
            if parent_id in id_to_agent:
                id_to_agent[parent_id].offspring.append(next_id)
        next_id += 1
    
    return next_id        
        
def pretty_print_agent(agent):
    """Prepares string for pretty-printing an entry from dead_agents to either console or logfile."""
    lines = []
    sex_label = 'm' if agent['sex'] == 0 else 'f'
    lines.append(f"ID {agent['id']}: born {agent['born']}, died {agent['died']} at ({agent['posx']}, {agent['posy']}), aged {agent['died']-agent['born']}, sex: {sex_label}")
    #genealogy+biography
    lines.append(f"  parent: {agent['parent']}, offspring: {len(agent['offspring'])} {agent['offspring']}, games played: {agent['games_played']}, avg score: {(agent['total_score']/agent['games_played']):.2f}")
    #tribal parameters
    lines.append(f"  identity: [{', '.join(f'{x:.2f}' for x in agent['identity'])}], ingroup preference: {agent['ingroup_preference']:.2f}")
    
    # Log sexual parameters only if ALLOW_SEXUAL is True
    if ALLOW_SEXUAL:
        bias_label = f", male bias: {agent['male_bias']:.2f}" if agent['sex'] == 0 else ''
        lines.append(f"  chivalry: {agent['chivalry']:.2f}, male investment: {agent['male_investment']:.2f}, mating display: {agent['mating_display']:.2f}, ladys choice: [{', '.join(f'{x:.2f}' for x in agent['ladys_choice'])}]{bias_label}")

    #Prisoner's dilemma strategy parameters. Print matrices: 2x2 lower triangular, then 6x6 lower triangular indented by 2 columns
    lines.append(f"  bias: {agent['cooperation_bias']:.2f}, linear: [{', '.join(f'{x:.2f}' for x in agent['linear_weights'])}]")    
    lines.append("  quadratic:")
    remote = agent['remote_weights']
    for i in range(2):
        row = [f"{remote[i][j]:>7.2f}" for j in range(i + 1)]
        lines.append("    " + " ".join(row))
    recent = agent['recent_weights']
    indent = "    " + " " * 16
    for i in range(6):
        row = [f"{recent[i][j]:>7.2f}" for j in range(i + 1)]
        lines.append(indent + " ".join(row))
    lines.append("-" * 50)
    return "\n".join(lines)
    
def write_agent_to_log(agent, file_path):
    with open(file_path, 'a') as f:
        f.write(pretty_print_agent(agent) + "\n")    

def log_dead_agents(dead_agents, file_path, last_logged_turn, clear_list=True):

    if not dead_agents:
        return last_logged_turn
    
    # Write only agents with died > last_logged_turn
    new_agents = [agent for agent in dead_agents if agent['died'] > last_logged_turn]
    for agent in new_agents:
        write_agent_to_log(agent, file_path)
    
    # Update last_logged_turn to the maximum died value
    if new_agents:
        last_logged_turn = max(agent['died'] for agent in new_agents)
    
    # Clear list if requested
    if clear_list:
        dead_agents.clear()
    
    return last_logged_turn
        
def log_simulation_stats(turn, agents, dead_agents, games_played, total_score, file_path):
    
    lines = []
    lines.append(f"Turns played: {turn}")
    lines.append(f"Final population: {len(agents)}")
    lines.append(f"Number of dead agents: {len(dead_agents)}\n")
    if dead_agents:
        most_offspring = max(dead_agents, key=lambda x: len(x['offspring']))
        lines.append(f"Agent with most offspring: ID {most_offspring['id']} with {len(most_offspring['offspring'])} offspring")
        highest_score = max(dead_agents, key=lambda x: x['total_score'])
        lines.append(f"Agent with highest total score: ID {highest_score['id']} with total score {highest_score['total_score']:.2f}")
        agents_with_games = [agent for agent in dead_agents if agent['games_played'] > 0]
        if agents_with_games:
            best_avg_score = max(agents_with_games, key=lambda x: x['total_score'] / x['games_played'])
            lines.append(f"Agent with best average score: ID {best_avg_score['id']} with average score {best_avg_score['total_score'] / best_avg_score['games_played']:.2f}")
    lines.append("-" * 50 + "\n")
    
    with open(file_path, 'a') as f:
        f.write("\n".join(lines))

#initialize pygame display
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Prisoner's Dilemma")
font = pygame.font.SysFont('arial', 24)

#founder identities (prepare four identities for founding agents in each corner)
TRIBE1 = [0 for x in range(IDENTITY_DIMENSIONS)]
TRIBE2 = [0 for x in range(IDENTITY_DIMENSIONS)]
TRIBE2[0]=10
TRIBE3 = [0 for x in range(IDENTITY_DIMENSIONS)]
TRIBE3[1]=1
TRIBE4 = [0 for x in range(IDENTITY_DIMENSIONS)]
TRIBE4[0]=10
TRIBE4[1]=10

#create empty agents array
next_id = 0
agents = []
# Initialize founding agents,e.g. two agents in opposite corners of the board
#breeding pair top-left
append_agent(
    posx=1, posy=1, id=next_id, born=0, sex=0,
    identity=TRIBE1, ingroup_preference=0.5,
    strategy="TFTT",
    sexual_selection="fascist",
    agents=agents
)
next_id += 1
append_agent(
    posx=1, posy=1, id=next_id, born=0, sex=1,
    identity=TRIBE1, ingroup_preference=0.5,
    strategy="TFTT",
    sexual_selection="fascist",
    agents=agents
)
next_id += 1

#breeding pair bottom-right
append_agent(
    posx=8, posy=8, id=next_id, born=0, sex=0,
    identity=TRIBE2, ingroup_preference=1.0,
    strategy="TFT",
    sexual_selection="balanced",
    agents=agents
)
next_id += 1
append_agent(
    posx=8, posy=8, id=next_id, born=0, sex=1,
    identity=TRIBE2, ingroup_preference=1.0,
    strategy="TFT",
    sexual_selection="balanced",
    agents=agents
)
next_id += 1


# initialize history dictionary and dead agents log
history_dict = defaultdict(lambda: (deque(maxlen=3), np.float16(0.0), np.float16(0.0), 0))

id_to_agent = {agent.id: agent for agent in agents}
dead_agents = []

#history graphs, pause button
population_history = []
cc_percentage_history = []
paused = False

def draw_visualization(agents, cc_games, dd_games, interactions, population_history, cc_percentage_history, turn, elapsed, paused, simulation_ended=False):
    screen.fill(WHITE)
    
    # Draw grid
    grid_label = font.render("Cell population / Cooperation (green) vs defection (red)", True, BLACK)
    screen.blit(grid_label, (GRID_OFFSET_X, GRID_OFFSET_Y - 30))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(GRID_OFFSET_X + x * CELL_SIZE, GRID_OFFSET_Y + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
            agent_count = sum(1 for agent in agents if agent.posx == x and agent.posy == y)
            C = (x, y)
            if interactions[C] > 0:
                coop_ratio = cc_games[C] / interactions[C] if cc_games[C] > 0 else 0
                defect_ratio = dd_games[C] / interactions[C] if dd_games[C] > 0 else 0
                color = (
                    min(255, int(255 * defect_ratio)),
                    min(255, int(255 * coop_ratio)),
                    0
                )
            else:
                color = (200, 200, 200)
            pygame.draw.rect(screen, color, rect.inflate(-2, -2))
            text = font.render(str(agent_count), True, BLACK)
            screen.blit(text, (GRID_OFFSET_X + x * CELL_SIZE + 20, GRID_OFFSET_Y + y * CELL_SIZE + 20))
    
    # Draw population and (C,C) ratio graphs
    graph_label = font.render("Population (blue) / (C,C) ratio (green)", True, BLACK)
    screen.blit(graph_label, (GRAPH_OFFSET_X, GRAPH_OFFSET_Y - 30))
    pygame.draw.rect(screen, BLACK, (GRAPH_OFFSET_X, GRAPH_OFFSET_Y, 500, 300), 1)
    if population_history:
        max_pop = max(population_history)
        for i in range(1, len(population_history)):
            x1 = GRAPH_OFFSET_X + (i-1) * 500 // NUM_TURNS
            y1 = GRAPH_OFFSET_Y + 300 - (population_history[i-1] * 300 // max_pop)
            x2 = GRAPH_OFFSET_X + i * 500 // NUM_TURNS
            y2 = GRAPH_OFFSET_Y + 300 - (population_history[i] * 300 // max_pop)
            pygame.draw.line(screen, BLUE, (x1, y1), (x2, y2), 2)
    if cc_percentage_history:
        for i in range(1, len(cc_percentage_history)):
            x1 = GRAPH_OFFSET_X + (i-1) * 500 // NUM_TURNS
            y1 = GRAPH_OFFSET_Y + 300 - (cc_percentage_history[i-1] * 300 // 100)
            x2 = GRAPH_OFFSET_X + i * 500 // NUM_TURNS
            y2 = GRAPH_OFFSET_Y + 300 - (cc_percentage_history[i] * 300 // 100)
            pygame.draw.line(screen, GREEN, (x1, y1), (x2, y2), 2)
    # Draw max pop label
    pop_label = font.render(f"Max Pop: {int(max_pop)}", True, BLACK)
    screen.blit(pop_label, (GRAPH_OFFSET_X + 350, GRAPH_OFFSET_Y))
    
    # Draw age and energy histograms
    hist_label = font.render("Agent age (blue) / score (red)", True, BLACK)
    screen.blit(hist_label, (HIST_OFFSET_X, HIST_OFFSET_Y - 30))
    pygame.draw.rect(screen, BLACK, (HIST_OFFSET_X, HIST_OFFSET_Y, 500, 300), 1)
    ages = [turn - agent.born for agent in agents]
    scores = [agent.energy for agent in agents]
    if ages and scores:
        max_age = max(ages) if ages else 1
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 1
        score_range = max_score - min_score if max_score != min_score else 1
        age_bins, age_edges = np.histogram(ages, bins=10, range=(0, max_age))
        score_bins, score_edges = np.histogram(scores, bins=10, range=(min_score, max_score))
        max_bin = max(max(age_bins), max(score_bins)) or 1
        
        # Plot age histogram (blue)
        age_points = []
        for i in range(len(age_bins)):
            x = HIST_OFFSET_X + (age_edges[i] / max_age) * 500
            y = HIST_OFFSET_Y + 300 - (age_bins[i] * 300 // max_bin)
            age_points.append((x, y))
        for i in range(1, len(age_points)):
            pygame.draw.line(screen, BLUE, age_points[i-1], age_points[i], 2)
        
        # Plot score histogram (red)
        score_points = []
        for i in range(len(score_bins)):
            x = HIST_OFFSET_X + ((score_edges[i] - min_score) / score_range) * 500
            y = HIST_OFFSET_Y + 300 - (score_bins[i] * 300 // max_bin)
            score_points.append((x, y))
        for i in range(1, len(score_points)):
            pygame.draw.line(screen, RED, score_points[i-1], score_points[i], 2)
        
        # Draw x-axis labels for energy (fixed 0 to 5, approximate)
        for i in range(6):
            label = font.render(str(i), True, BLACK)
            screen.blit(label, (HIST_OFFSET_X + i * 100 - 10, HIST_OFFSET_Y + 310))
        
        # Draw max age label
        age_label = font.render(f"Max Age: {int(max_age)}", True, BLACK)
        screen.blit(age_label, (HIST_OFFSET_X + 400, HIST_OFFSET_Y + 340))

	#xenophobia histogram
        pygame.draw.rect(screen, BLACK, (GRAPH2_OFFSET_X, GRAPH2_OFFSET_Y, 500, 300), 1)
        hist_label = font.render("xenophobia/nepotism", True, BLACK)
        screen.blit(hist_label, (GRAPH2_OFFSET_X, GRAPH2_OFFSET_Y - 30))
        xenophobia = [agent.ingroup_preference for agent in agents]
    if xenophobia:
        xe_bins, xe_edges = np.histogram(xenophobia, bins=10, range=(0, 1))
        max_bin = max(xe_bins) or 1        
        # Plot  histogram 
        xe_points = []
        for i in range(len(xe_bins)):
            x = GRAPH2_OFFSET_X + (xe_edges[i]) * 500
            y = GRAPH2_OFFSET_Y + 300 - (xe_bins[i] * 300 // max_bin)
            xe_points.append((x, y))
        for i in range(1, len(xe_points)):
            pygame.draw.line(screen, GREEN, xe_points[i-1], xe_points[i], 2)

	#identity histogram: similarity to TRIBE1 and TRIBE2 (eventually: four founder tribes?)
        pygame.draw.rect(screen, BLACK, (HIST2_OFFSET_X, HIST2_OFFSET_Y, 500, 300), 1)        
        hist_label = font.render("Identity tribe1: (blue) / tribe2 (red)", True, BLACK)
        screen.blit(hist_label, (HIST2_OFFSET_X, HIST2_OFFSET_Y - 30))
        distances1 = [np.sum((agent.identity - TRIBE1)**2)/IDENTITY_DIMENSIONS for agent in agents] #euclidian distance squared, in range [0,1]
        distances2 = [np.sum((agent.identity - TRIBE2)**2)/IDENTITY_DIMENSIONS for agent in agents]
    if distances1 and distances2:
        dist1_bins, dist1_edges = np.histogram(distances1, bins=10, range=(0, 1))
        dist2_bins, dist2_edges = np.histogram(distances2, bins=10, range=(0, 1))
        max_bin = max(max(dist1_bins), max(dist2_bins)) or 1        
        # Plot identity histogram 
        dist1_points = []
        for i in range(len(dist1_bins)):
            x = HIST2_OFFSET_X + (dist1_edges[i]) * 500
            y = HIST2_OFFSET_Y + 300 - (dist1_bins[i] * 300 // max_bin)
            dist1_points.append((x, y))
        for i in range(1, len(dist1_points)):
            pygame.draw.line(screen, BLUE, dist1_points[i-1], dist1_points[i], 2)
        dist2_points = []
        for i in range(len(dist2_bins)):
            x = HIST2_OFFSET_X + (dist2_edges[i]) * 500
            y = HIST2_OFFSET_Y + 300 - (dist2_bins[i] * 300 // max_bin)
            dist2_points.append((x, y))
        for i in range(1, len(dist2_points)):
            pygame.draw.line(screen, RED, dist2_points[i-1], dist2_points[i], 2)


    # Draw numeric data
    total_interactions = sum(interactions.values())
    ns_per_int = elapsed/total_interactions*1e+9 if total_interactions > 0 else 0
    total_cc = sum(cc_games.values())
    total_dd = sum(dd_games.values())
    total_cd = total_interactions - total_cc - total_dd
    cc_percent = (total_cc / total_interactions * 100) if total_interactions > 0 else 0
    dd_percent = (total_dd / total_interactions * 100) if total_interactions > 0 else 0
    cd_percent = (total_cd / total_interactions * 100) if total_interactions > 0 else 0
    data_texts = [
        f"Turn: {turn}",
        f"Population: {len(agents)}; ancestor log: {len(dead_agents)}",
        f"Interactions: {total_interactions}",
        f"(C,C): {cc_percent:.1f}%",
        f"(D,D): {dd_percent:.1f}%",
        f"(C,D): {cd_percent:.1f}%",
        f"elapsed: {(elapsed*1000):.0f} ms (per interaction: {ns_per_int/1000:.1f} μs)"
    ]
    for i, text in enumerate(data_texts):
        rendered = font.render(text, True, BLACK)
        screen.blit(rendered, (DATA_OFFSET_X, DATA_OFFSET_Y + i * 30))
    
    # Draw pause/exit button
    button_rect = pygame.Rect(BUTTON_OFFSET_X, BUTTON_OFFSET_Y, 100, 40)
    button_color = GRAY if paused and not simulation_ended else YELLOW
    pygame.draw.rect(screen, button_color, button_rect)
    button_text = "Exit" if simulation_ended else "Pause" if not paused else "Resume"
    text = font.render(button_text, True, BLACK)
    text_rect = text.get_rect(center=button_rect.center)
    screen.blit(text, text_rect)
    
    pygame.display.flip()

def handle_button_click(pos, paused, simulation_ended):
    button_rect = pygame.Rect(BUTTON_OFFSET_X, BUTTON_OFFSET_Y, 100, 40)
    if button_rect.collidepoint(pos):
        if simulation_ended:
            return True, paused  # Exit
        else:
            return False, not paused  # Toggle pause
    return False, paused

###############################################################################################################
# main loop

async def main():
    global agents, next_id, history_dict, id_to_agent, population_history, cc_percentage_history, turn, paused, timestamp
    turn = 0
    last_logged_turn=0
    if LOG_TO_FILE:
        with open(logfile_path, 'w') as f:
            f.write("Dead Agents Log\n")
            f.write("=" * 15 + "\n")

    while turn < NUM_TURNS:
        timestamp = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                should_exit, paused = handle_button_click(event.pos, paused, False)
                if should_exit:
                    return        
        if paused:
            draw_visualization(agents, cc_games, dd_games, interactions, population_history, cc_percentage_history, turn, 0, paused)
            await asyncio.sleep(1.0 / FPS)
            continue

#statistics for this turn, to be reset each turn
        cc_games = defaultdict(int)
        dd_games = defaultdict(int)
        interactions = defaultdict(int)

    #INTERACTION_DEPTH: interactions per pair per turn, in principle could be set to "full game" value (50, 100 or 200) at the cost of extremely long runtimes

    #main game loop overview:
    #step1: consumption+death (agents consume energy, starved agents die off)        
    #step2: interaction phase (interacting pairs processed in parallel, call to run_interaction and thence process_chunk)
    # step2a: sexual reproduction is also handled in run_interaction since we were already going over interacting_pairs
    #step3: energy distribution based on interaction outcomes
    #step4: asexual reproduction
    #step5: migration: cells may migrate to lower population cells

    # Step 1: Consumption & Death
        for agent in agents:
            agent.energy -= CONSUMPTION
            agent.energy -= AGING_PENALTY*(turn-agent.born)
            if turn < DEBUG_TURNS:
                print(f"Turn {turn}, Agent {agent.id} at ({agent.posx},{agent.posy}): Energy after consumption = {agent.energy:.2f}")

        dead_this_turn = [agent for agent in agents if agent.energy <= 0]
        deaths = len(dead_this_turn)
        for agent in dead_this_turn:
            dead_agents.append(agent_snapshot(agent, games_played, total_score))
	    #some housekeeping (delete no longer needed array entries)
            del games_played[agent.id]
            del total_score[agent.id]
            del interactions_per_agent[agent.id]
            keys_to_remove = [key for key in history_dict if agent.id in key]
            for key in keys_to_remove:
                del history_dict[key]

        agents = [agent for agent in agents if agent.energy > 0]
        id_to_agent = {agent.id: agent for agent in agents}
        if LOG_TO_FILE and turn % LOG_INTERVAL == 0:
            last_logged_turn = log_dead_agents(dead_agents, logfile_path, last_logged_turn)

    # census: update cell populations and agent mapping
        population = defaultdict(int)
        cell_to_agents = defaultdict(list)
        female_pop = 0
        for agent in agents:
            cell = (agent.posx, agent.posy)
            population[cell] += 1
            cell_to_agents[cell].append(agent)
            female_pop += agent.sex

    # Step 2: Interaction Phase
    
        interacting_pairs = []
        interaction_payouts = []
        interactions_per_agent = defaultdict(int)
        
        for cell in cell_to_agents:
            agents_in_cell = cell_to_agents[cell]
            for i, A in enumerate(agents_in_cell):
                for B in agents_in_cell[i+1:]:
                    interacting_pairs.append((A, B))
                    interactions_per_agent[A.id] += 1
                    interactions_per_agent[B.id] += 1
                    if turn < DEBUG_TURNS:
                        print(f"Turn {turn}, Interaction: Agent {A.id} at ({A.posx},{A.posy}) vs Agent {B.id} at ({B.posx},{B.posy})")
            x, y = cell
            adj_cells = [(x+dx, y+dy) for dx in [-1,0,1] for dy in [-1,0,1] if (dx,dy) != (0,0) and 0 <= x+dx < 10 and 0 <= y+dy < 10]
            for adj_cell in adj_cells:
                for A in agents_in_cell:
                    for B in cell_to_agents.get(adj_cell, []):
                        interacting_pairs.append((A, B))
                        interactions_per_agent[A.id] += 1
                        interactions_per_agent[B.id] += 1
                        if turn < DEBUG_TURNS:
                            print(f"Turn {turn}, Interaction: Agent {A.id} at ({A.posx},{A.posy}) vs Agent {B.id} at ({B.posx},{B.posy})")

        #parallelized interaction loop:
        if len(interacting_pairs) > 0:
            next_id = run_interactions(interacting_pairs, history_dict, interaction_payouts, interactions, total_score, games_played, cc_games, dd_games, next_id)

    # Step 3: Energy Distribution
        energy_change = defaultdict(float)
        for cell, agents_in_cell in cell_to_agents.items():
            if interactions[cell] == 0:
                energy_per_agent = EXTRACTION_BASE / population[cell]
                for agent in agents_in_cell:
                    energy_change[agent.id] += energy_per_agent
                    if turn < DEBUG_TURNS:
                        print(f"Turn {turn}, Agent {agent.id} at {cell}: Gained {energy_per_agent:.2f} (no interactions)")
            else:
                energy_per_agent = EXTRACTION_BASE / population[cell]
                for agent in agents_in_cell:
                    n = interactions_per_agent[agent.id]
                    dE = energy_per_agent / (n + 1) 
                    energy_change[agent.id] += dE

    # Apply payouts
        for aid1, aid2, score1, score2, mod in interaction_payouts:
            n1 = interactions_per_agent[aid1]
            dE1 = (EXTRACTION_BASE / population[(id_to_agent[aid1].posx, id_to_agent[aid1].posy)]) / (n1 + 1)
            n2 = interactions_per_agent[aid2]
            dE2 = (EXTRACTION_BASE / population[(id_to_agent[aid2].posx, id_to_agent[aid2].posy)]) / (n2 + 1)

#score1, score2 are accumulated scores after INTERACTION_DEPTH iterations
#mod (modifier) contains bonus/penalty for (C,C) and (D,D) interaction. Since these are symmetric, only one value is needed
#agents play for energy amount (dE1+dE2), which is distributed proportionally to their score
#important to keep in mind that agents with more neighbors have lower stakes in each interaction, so their risk is spread out more. This might incentivize higher risk strategies in higher population density areas

            energy_change[aid1] += (dE1+dE2)*score1/(score1+score2)*mod
            energy_change[aid2] += (dE1+dE2)*score2/(score1+score2)*mod
            if turn < DEBUG_TURNS:
                print(f"Turn {turn}, Payout: Agent {aid1} gets {(dE1+dE2)*score1/(score1+score2)*mod:.2f}, Agent {aid2} gets keeps {(dE1+dE2)*score1/(score1+score2)*mod:.2f}")

#note: we used to check for interaction outcomes explicitly and apply cooperation bonus here, this is decommissioned to accommodate INTERACTION_DEPTH, cooperation bonus is now applied in process_chunk

        for aid, change in energy_change.items():
            id_to_agent[aid].energy += change
            if turn < DEBUG_TURNS:
                print(f"Turn {turn}, Agent {aid}: Final energy = {id_to_agent[aid].energy:.2f}")
                print(f"Turn {turn}, Agent {aid}: Linear weights = {id_to_agent[aid].linear_weights.tolist()}")
                print(f"Turn {turn}, Agent {aid}: Prop weights = {id_to_agent[aid].remote_weights.tolist()}")
                print(f"Turn {turn}, Agent {aid}: History weights = {id_to_agent[aid].recent_weights.tolist()}")
                print(f"Turn {turn}, Agent {aid}: Identity = {id_to_agent[aid].identity.tolist()}")
                print(f"Turn {turn}, Agent {aid}: Ingroup preference = {id_to_agent[aid].ingroup_preference:.2f}")

    # Step 4: Asexual reproduction
        births = 0
        if ALLOW_ASEXUAL:
            new_agents = []
            for agent in agents[:]:
                while agent.energy > ASEXUAL_REPRODUCTION_THRESHOLD + ASEXUAL_REPRODUCTION_COST:
                    possible_cells = [(agent.posx + dx, agent.posy + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if 0 <= agent.posx + dx < 10 and 0 <= agent.posy + dy < 10]
                    new_pos = random.choice(possible_cells)
                    mutated_cooperation_bias, mutated_linear, mutated_remote, mutated_recent, mutated_identity, mutated_ingroup_preference, mutated_male_bias = mutate_genome(
                        agent.cooperation_bias, agent.linear_weights, agent.remote_weights, agent.recent_weights, agent.identity, agent.ingroup_preference, agent.male_bias, agent.male_investment, agent.mating_display, agent.chivalry, agent.ladys_choice)
                    new_sex=random.randint(0,1)
                    new_male_bias = mutated_male_bias if new_sex == 0 else 0.0
                    new_agent = Agent(
                        posx=new_pos[0],
                        posy=new_pos[1],
                        energy=OFFSPRING_ENERGY,
                        id=next_id,
                        parent=agent.id,
                        born=turn,
                        cooperation_bias=mutated_cooperation_bias,
                        linear_weights=mutated_linear,
                        remote_weights=mutated_remote,
                        recent_weights=mutated_recent,
                        identity=mutated_identity,
                        ingroup_preference=mutated_ingroup_preference,
                        sex=new_sex,
                        male_bias=new_male_bias
                    )
                    next_id += 1
                    births += 1
                    new_agents.append(new_agent)
                    agent.offspring.append(new_agent.id)
                    agent.energy -= (OFFSPRING_ENERGY + ASEXUAL_REPRODUCTION_COST)
                    if turn < DEBUG_TURNS:
                        print(f"Turn {turn}, Agent {agent.id}: Reproduced, new agent {new_agent.id} at ({new_pos[0]},{new_pos[1]}), parent energy = {agent.energy:.2f}")
                    if turn < DEBUG_TURNS and games_played[agent.id] > 0 and total_score[agent.id] / games_played[agent.id] > 3.0:
                        print(f"Turn {turn}, Agent {agent.id}: Reproduction success, Energy = {agent.energy:.2f}, New cell = ({new_pos[0]},{new_pos[1]})")

            agents.extend(new_agents)
            id_to_agent.update({agent.id: agent for agent in new_agents})

    # Step 5: Migration
    # comment: migration mechanic may be completely irrelevant for simulation dynamics, expansion is dominated by reproduction, perhaps disallow spawning of offspring into neighboring cells so migration becomes necessary for expansion, alternatively just get rid of migration altogether
    
        migrations = []
        #decide on all migrations before applying:
        for cell in cell_to_agents:
            agents_in_cell = cell_to_agents[cell]
            for A in agents_in_cell:
                current_pop = population[(A.posx, A.posy)]
                possible_cells = [(A.posx + dx, A.posy + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if 0 <= A.posx + dx < 10 and 0 <= A.posy + dy < 10]
                lower_pop_cells = [cell for cell in possible_cells if population[cell] < current_pop]
                if lower_pop_cells and random.random() < MIGRATION_RATE and (A.energy > CONSUMPTION + MIGRATION_COST):
                    new_pos = random.choice(lower_pop_cells)
                    migrations.append((A, new_pos[0], new_pos[1]))

        #apply all migrations at simultaneously    
        for agent, new_posx, new_posy in migrations:
            agent.posx, agent.posy = new_posx, new_posy
            agent.energy -= MIGRATION_COST
            if turn < DEBUG_TURNS and games_played[agent.id] > 0 and total_score[agent.id] / games_played[agent.id] > 3.0:
                print(f"Turn {turn}, Agent {agent.id}: Migrated to ({new_posx},{new_posy}), Energy after migration = {agent.energy:.2f}")

    # Update population and agent mapping after migrations
    # comment: we did this already at the start of the game loop, what a waste
    # this is recomputed here just for accurate turn statistics calculation below. 
    # this could be simply omitted at the cost of turn statistics lagging behind half a turn or so.
        population = defaultdict(int)
        cell_to_agents = defaultdict(list)
        for agent in agents:
            cell = (agent.posx, agent.posy)
            population[cell] += 1
            cell_to_agents[cell].append(agent)

    # Compute turn statistics
        total_interactions = sum(interactions[cell] for cell in interactions) // 2
        total_cc = sum(cc_games[cell] for cell in cc_games) // 2
        total_dd = sum(dd_games[cell] for cell in dd_games) // 2
        total_cd = total_interactions - total_cc - total_dd
        scores = [total_score[agent.id] / games_played[agent.id] for agent in agents if games_played[agent.id] > 0]
        avg_score = np.mean(scores) if scores else 0.0
        score_std = np.std(scores) if scores else 0.0
        
    # Update histories
        population_history.append(len(agents))
        cc_percent = (total_cc / total_interactions * 100) if total_interactions > 0 else 0
        cc_percentage_history.append(cc_percent)

    # Update visualization every DRAW_INTERVAL-th turn
        if turn % DRAW_INTERVAL == 0:
            draw_visualization(agents, cc_games, dd_games, interactions, population_history, cc_percentage_history, turn,  (time.time()-timestamp), paused)

    # Print progress
    # somewhat outdated, to be reviewed ("births" no longer accurate in case of sexual reproduction)
        if (turn % PRINT_INTERVAL == 0 or turn < DEBUG_TURNS) and total_interactions > 0:
            print(f"Turn {turn}: Population = {len(agents)}, Births = {births}, Deaths = {deaths}, Sex ratio {female_pop/len(agents):.2f}, "
                  f"Interactions = {total_interactions}, (C,C) = {(total_cc/total_interactions):.2f}, (D,D) = {(total_dd/total_interactions):.2f}, (C,D) = {(total_cd/total_interactions):.2f}, "
                  f"Avg Score = {avg_score:.2f}, Score Std = {score_std:.2f}")
            if turn < DEBUG_TURNS:
                print(f"Outcome check: (C,C) + (D,D) + (C,D) = {total_cc + total_dd + total_cd}, Interactions = {total_interactions}")
                print("Agent positions:", [(agent.id, agent.posx, agent.posy, agent.energy) for agent in agents])

        turn += 1
        await asyncio.sleep(1.0 / FPS)

#end of game loop. exits after NUM_TURN is reached, could also run indefinitely and wait for button press to end simulation instead. 
# maximum number of turns is mainly there to avoid indefinite pileup of log data (dead_agents), instead we could just stop logging dead agents past a certain number.
#########################################################################################################################################

# Final statistics
    print("Final population:", len(agents))
    print("Number of dead agents:", len(dead_agents))
    if dead_agents:
        most_offspring = max(dead_agents, key=lambda x: len(x['offspring']))
        print(f"Agent with most offspring: ID {most_offspring['id']} with {len(most_offspring['offspring'])} offspring")
        highest_score = max(dead_agents, key=lambda x: x['total_score'])
        print(f"Agent with highest total score: ID {highest_score['id']} with total score {highest_score['total_score']}")
        agents_with_games = [agent for agent in dead_agents if agent['games_played'] > 0]
        if agents_with_games:
            best_avg_score = max(agents_with_games, key=lambda x: x['total_score'] / x['games_played'])
            print(f"Agent with best average score: ID {best_avg_score['id']} with average score {best_avg_score['total_score'] / best_avg_score['games_played']:.2f}")


#this is a placeholder for a more sophisticated analysis of what has happened. The idea would be to analyze which strategies were successful, when they arose, and possibly if there were epochs or eras during the run during which certain strategies dominated and when and how these eras ended, perhaps cycles of chaos and order, etc.

        print("Listing some interesting dead agents:")
        for dead_agent in dead_agents:
            if (len(dead_agent['offspring']) > (len(most_offspring['offspring']) * 0.9) or (dead_agent['games_played'] > 0 and (dead_agent['total_score'] / dead_agent['games_played']) > (best_avg_score['total_score'] / best_avg_score['games_played'] * 0.9)) or (dead_agent['died'] - dead_agent['born']) > (NUM_TURNS / 2)):
                print (pretty_print_agent(dead_agent))

    if LOG_TO_FILE:
        last_logged_turn = log_dead_agents(dead_agents, logfile_path, last_logged_turn)
        log_simulation_stats(turn, agents, dead_agents, games_played, total_score, logfile_path)

    # Post-simulation loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                should_exit, _ = handle_button_click(event.pos, paused, True)
                if should_exit:
                    return
        
        draw_visualization(agents, cc_games, dd_games, interactions, population_history, cc_percentage_history, turn,  0, paused, True)
        await asyncio.sleep(1.0 / FPS)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())

