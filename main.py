import numpy as np
import random
from collections import defaultdict, deque
from multiprocessing import Process, Queue
import queue  # Added for queue.Empty
import pygame
import psutil #just to ask for number of physical cores
import platform
import time
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
from tkinter import filedialog
import sys
import os

from sample_strategies import sample_strategies, sexual_strategies

# Global initialization
turn = 0
FPS = 20 

NUM_CORES = psutil.cpu_count(logical=False)
if NUM_CORES is None:
    NUM_CORES = 8 # observed about 4.5x speedup for 8 cores

#the game
NUM_TURNS = 400
INTERACTION_DEPTH = 3 #interactions per turn
GRID_SIZE = 10

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
    else:
        mutated_male_bias = 0
        mutated_male_investment = 0
        mutated_mating_display = 0
        mutated_chivalry=0
        mutated_ladys_choice=(0,0,0,0)

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
            kinship = 1/(np.sum((male.identity - mother.identity) ** 2) / IDENTITY_DIMENSIONS +1)   
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

# Tribe mapping for agent identities
# currently set up so that "10" is a large distance, and "1" is a moderate or small one. Max distance is sqrt(500) = 22.3. Founding tribes start at distances between 10 and 17.3, so "ethnically distinct"
TRIBE_MAP = {
    "A": [0] * IDENTITY_DIMENSIONS,
    "B": [10] + [0] * (IDENTITY_DIMENSIONS - 1),
    "C": [0, 10] + [0] * (IDENTITY_DIMENSIONS - 2),
    "D": [10, 10] + [0] * (IDENTITY_DIMENSIONS - 2),
    "E": [10, 10, 10] + [0] * (IDENTITY_DIMENSIONS - 3)
}
# Founder positions for spawning agents
gr_lo = int((GRID_SIZE-1)*.1+.5)  #1, but derive from GRID_SIZE in case this is made dynamic
gr_hi = int((GRID_SIZE-1)*.9+.5)  #8
gr_mid = int((GRID_SIZE-1)*.5+.5) #5
FOUNDER_POSITIONS = [(gr_lo, gr_lo), (gr_hi, gr_hi), (gr_lo, gr_hi), (gr_hi, gr_lo), (gr_mid, gr_mid)]

# Parameter to constant mapping
PARAM_TO_CONSTANT = {
    "num_turns": ("NUM_TURNS", int),
    "interaction_depth": ("INTERACTION_DEPTH", int),
    "base_energy": ("EXTRACTION_BASE", int),
    "consumption": ("CONSUMPTION", int),
    "migration_rate": ("MIGRATION_RATE", float),
    "migration_cost": ("MIGRATION_COST", int),
    "aging_penalty": ("AGING_PENALTY", float),
    "mutation_rate": ("MUTATION_RATE", float),
    "mutation_rate_tribal": ("MUTATION_RATE_IDENTITY", float),
    "mutation_rate_sexual": ("MUTATION_RATE_SEXUAL", float),
    "log_to_file": ("LOG_TO_FILE", bool),
    "logfile": ("logfile_path", str),
    "log_turns": ("LOG_INTERVAL", int),
    "allow_asexual": ("ALLOW_ASEXUAL", bool),
    "allow_sexual": ("ALLOW_SEXUAL", bool)
}

def update_constants(params):
    """Update global constants from params dictionary."""
    global NUM_TURNS, INTERACTION_DEPTH, EXTRACTION_BASE, CONSUMPTION, MIGRATION_RATE, MIGRATION_COST
    global AGING_PENALTY, MUTATION_RATE, MUTATION_RATE_IDENTITY, MUTATION_RATE_SEXUAL
    global LOG_TO_FILE, logfile_path, LOG_INTERVAL, ALLOW_ASEXUAL, ALLOW_SEXUAL
    for param_key, (const_name, const_type) in PARAM_TO_CONSTANT.items():
        try:
            globals()[const_name] = const_type(params[param_key])
        except (KeyError, ValueError) as e:
            print(f"Error updating {const_name}: {e}")
#    print(f"Updated constants: NUM_TURNS={NUM_TURNS}, INTERACTION_DEPTH={INTERACTION_DEPTH}, EXTRACTION_BASE={EXTRACTION_BASE}, CONSUMPTION={CONSUMPTION}, MIGRATION_RATE={MIGRATION_RATE}, MIGRATION_COST={MIGRATION_COST}, AGING_PENALTY={AGING_PENALTY}, MUTATION_RATE={MUTATION_RATE}, MUTATION_RATE_IDENTITY={MUTATION_RATE_IDENTITY}, MUTATION_RATE_SEXUAL={MUTATION_RATE_SEXUAL}, LOG_TO_FILE={LOG_TO_FILE}, logfile_path={logfile_path}, LOG_INTERVAL={LOG_INTERVAL}, ALLOW_ASEXUAL={ALLOW_ASEXUAL}, ALLOW_SEXUAL={ALLOW_SEXUAL}")

agents=[]
next_id=0

# initialize history dictionary and dead agents log
history_dict = defaultdict(lambda: (deque(maxlen=3), np.float16(0.0), np.float16(0.0), 0))

id_to_agent = {agent.id: agent for agent in agents}
dead_agents = []

#history graphs, pause button
population_history = []
cc_percentage_history = []
female_ratio_history = []
paused = False

# pygame visualization ###################################################################
# Visualization constants
BASE_WIDTH, BASE_HEIGHT = 1920, 1080  # Base canvas size
ASPECT_RATIO = BASE_WIDTH / BASE_HEIGHT  # 1.78:1
MIN_WIDTH, MAX_WIDTH = 600, 3000  # Canvas width constraints
CELL_SIZE = int(800 / GRID_SIZE)  # Dynamic cell size
HISTO_BINS = 20  # Histogram bin count
SCATTERPLOT_ENABLED = True  # Toggle scatterplot
GRID_OFFSET_X, GRID_OFFSET_Y = 40, 50
GRAPH1_OFFSET_X, GRAPH1_OFFSET_Y = 880, 50  # Population/CC ratio/Female ratio
GRAPH2_OFFSET_X, GRAPH2_OFFSET_Y = 1400, 50  # Reproduction/Sexual Selection
GRAPH3_OFFSET_X, GRAPH3_OFFSET_Y = 880, 400  # Age/Energy/Score
GRAPH4_OFFSET_X, GRAPH4_OFFSET_Y = 1400, 400  # Tribal (Identity/Xenophobia)
DATA_OFFSET_X, DATA_OFFSET_Y = 900, 750
BUTTON_OFFSET_X, BUTTON_OFFSET_Y = 900, 980
WHITE = (250, 250, 250)
BLACK = (5, 5, 5)
GREEN = (20, 255, 30)
RED = (255, 20, 30)
BLUE = (50, 50, 255)
GRAY = (150, 150, 150)
YELLOW = (250, 250, 10)
PINK = (255, 105, 180)
LIGHT_PINK = (255, 182, 193)

def draw_visualization(screen, agents, cc_games, dd_games, interactions, population_history, cc_percentage_history, female_ratio_history, total_score, population, font, turn, elapsed, paused, text_cache, simulation_ended=False):
    """This uses pygame to display simulation state parameters at the end of a turn. With increasing game complexity and increasing sophistication of the display features, this is destined to becoming a bit of a resource hog, both in terms of performance and of developmental effort. Currently, the display should take about 5% of processing time if drawn every turn. It is always possible to set DRAW_INTERVAL to a value >1 and only draw the display every couple of turns."""
    # Measure total draw time
    start_time = time.perf_counter_ns()
    #sample runtime (as of 18 Jun 25):
    #Draw time: 10.9 ms (Canvas: 2.0, Grid: 4.0, Graph1: 0.9, Graph2: 0.3, Graph3: 0.5, Graph4: 1.6, Scatter Gather: 0.9, Scatter Draw: 0.5, Data: 0.2, Button: 0.0, Flip: 1.3)
    #This means drawing the "grid" (board) is the bottleneck, taking ~35% of total time, or, since canvas+flip are fixed cost of drawing the thing at all, up to ~55% of time spent actually drawing things.
    #Most of this cost is associated with using a full font and scaling it dynamically to draw the population numbers on the cells, a 100 times over. This could probably be much improved using pre-rendered numerals and just slapping a prepared bit pattern from a 10-dim array on there. But this would create headaches with window scaling. 
    #Anyway, note to self, pre-rendering numerals is the first thing to look into if performance becomes an issue that can no longer be ignored.
    #Also, consider looking into pygame.HWSURFACE:
    # #screen = pygame.display.set_mode((1920, 1080), pygame.HWSURFACE | pygame.DOUBLEBUF)
    # #renderer = pygame._sdl2.video.Renderer.from_window(pygame.display.get_window())
    # #texture = renderer.create_texture(pygame.SDL_PIXELFORMAT_RGBA8888, 1920, 1080)
    # this could cut flip from say 1.5 ms to 1.0 ms by giving the bitmap to the GPU as a "texture" to draw. Not worth the effort at this point, especially since the feature is "experimental".
    
    # Canvas setup
    canvas_start = time.perf_counter_ns()
    window_width, window_height = screen.get_size()
    window_aspect = window_width / window_height
    if window_aspect > ASPECT_RATIO:
        canvas_width = int(window_height * ASPECT_RATIO)
        canvas_height = window_height
    else:
        canvas_width = window_width
        canvas_height = int(window_width / ASPECT_RATIO)
    canvas_width = max(MIN_WIDTH, min(MAX_WIDTH, canvas_width))
    canvas_height = int(canvas_width / ASPECT_RATIO)
    scale = canvas_width / BASE_WIDTH
    canvas_x = (window_width - canvas_width) // 2
    canvas_y = (window_height - canvas_height) // 2
    canvas_rect = pygame.Rect(canvas_x, canvas_y, canvas_width, canvas_height)
    screen.fill(WHITE, canvas_rect)
    s_grid_offset_x = int(GRID_OFFSET_X * scale) + canvas_x
    s_grid_offset_y = int(GRID_OFFSET_Y * scale) + canvas_y
    s_cell_size = int(CELL_SIZE * scale)
    s_graph1_offset_x = int(GRAPH1_OFFSET_X * scale) + canvas_x
    s_graph1_offset_y = int(GRAPH1_OFFSET_Y * scale) + canvas_y
    s_graph2_offset_x = int(GRAPH2_OFFSET_X * scale) + canvas_x
    s_graph2_offset_y = int(GRAPH2_OFFSET_Y * scale) + canvas_y
    s_graph3_offset_x = int(GRAPH3_OFFSET_X * scale) + canvas_x
    s_graph3_offset_y = int(GRAPH3_OFFSET_Y * scale) + canvas_y
    s_graph4_offset_x = int(GRAPH4_OFFSET_X * scale) + canvas_x
    s_graph4_offset_y = int(GRAPH4_OFFSET_Y * scale) + canvas_y
    s_data_offset_x = int(DATA_OFFSET_X * scale) + canvas_x
    s_data_offset_y = int(DATA_OFFSET_Y * scale) + canvas_y
    s_button_offset_x = int(BUTTON_OFFSET_X * scale) + canvas_x
    s_button_offset_y = int(BUTTON_OFFSET_Y * scale) + canvas_y
    canvas_time = (time.perf_counter_ns() - canvas_start) / 1000
    
    # Draw grid
    grid_start = time.perf_counter_ns()
    grid_surface = pygame.Surface((s_cell_size * GRID_SIZE, s_cell_size * GRID_SIZE), pygame.SRCALPHA)
    grid_label = font.render("Cell population / Cooperation (green) vs defection (red)", True, BLACK)
    grid_label = pygame.transform.smoothscale_by(grid_label, scale)
    screen.blit(grid_label, (s_grid_offset_x, s_grid_offset_y - int(30 * scale)))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(x * s_cell_size, y * s_cell_size, s_cell_size, s_cell_size)
            pygame.draw.rect(grid_surface, BLACK, rect, 1)
            agent_count = population.get((x, y), 0)
            C = (x, y)
            if interactions[C] > 0:
                coop_ratio = cc_games[C] / interactions[C] if cc_games[C] > 0 else 0
                defect_ratio = dd_games[C] / interactions[C] if dd_games[C] > 0 else 0
                cd_count = interactions[C] - cc_games[C] - dd_games[C]
                cd_ratio = cd_count / interactions[C] if cd_count > 0 else 0
                base_r = 255 * defect_ratio
                base_g = 255 * coop_ratio
                base_b = 0
                cd_r, cd_g, cd_b = 200, 200, 10
                r = int(base_r * (1 - cd_ratio) + cd_r * cd_ratio)
                g = int(base_g * (1 - cd_ratio) + cd_g * cd_ratio)
                b = int(base_b * (1 - cd_ratio) + cd_b * cd_ratio)
                color = (r, g, b)
            else:
                color = (200, 200, 200)
            pygame.draw.rect(grid_surface, color, rect.inflate(-int(2 * scale), -int(2 * scale)))
            if agent_count <= 10:  # Cache common counts
                key = f"count_{agent_count}_{int(24 * scale)}"
                if key not in text_cache:
                    text = font.render(str(agent_count), True, BLACK)
                    text = pygame.transform.smoothscale_by(text, scale)
                    text_cache[key] = text
                text = text_cache[key]
            else:
                text = font.render(str(agent_count), True, BLACK)
                text = pygame.transform.smoothscale_by(text, scale)
            text_pos = (x * s_cell_size + int(20 * scale), y * s_cell_size + int(20 * scale))
            grid_surface.blit(text, text_pos)
    screen.blit(grid_surface, (s_grid_offset_x, s_grid_offset_y))
    grid_time = (time.perf_counter_ns() - grid_start) / 1000
    
    # Draw population, cc ratio, and female ratio (graph1)
    graph1_start = time.perf_counter_ns()
    graph_label = font.render("Population / CC ratio / Female ratio", True, BLACK)
    graph_label = pygame.transform.smoothscale_by(graph_label, scale)
    screen.blit(graph_label, (s_graph1_offset_x, s_graph1_offset_y - int(30 * scale)))
    pygame.draw.rect(screen, BLACK, (s_graph1_offset_x, s_graph1_offset_y, int(500 * scale), int(300 * scale)), 1)
    if population_history:
        max_entries = 200
        if len(population_history) > max_entries:
            indices = np.linspace(0, len(population_history)-1, max_entries, dtype=int)
            draw_population = [population_history[i] for i in indices]
        else:
            draw_population = population_history
        max_pop = max(draw_population)
        for i in range(1, len(draw_population)):
            x1 = s_graph1_offset_x + (i-1) * int(500 * scale) // max_entries
            y1 = s_graph1_offset_y + int(300 * scale) - (draw_population[i-1] * int(300 * scale) // max_pop)
            x2 = s_graph1_offset_x + i * int(500 * scale) // max_entries
            y2 = s_graph1_offset_y + int(300 * scale) - (draw_population[i] * int(300 * scale) // max_pop)
            pygame.draw.line(screen, BLUE, (x1, y1), (x2, y2), int(2 * scale))
    if cc_percentage_history:
        if len(cc_percentage_history) > max_entries:
            indices = np.linspace(0, len(cc_percentage_history)-1, max_entries, dtype=int)
            draw_cc = [cc_percentage_history[i] for i in indices]
        else:
            draw_cc = cc_percentage_history
        for i in range(1, len(draw_cc)):
            x1 = s_graph1_offset_x + (i-1) * int(500 * scale) // max_entries
            y1 = s_graph1_offset_y + int(300 * scale) - (draw_cc[i-1] * int(300 * scale) // 100)
            x2 = s_graph1_offset_x + i * int(500 * scale) // max_entries
            y2 = s_graph1_offset_y + int(300 * scale) - (draw_cc[i] * int(300 * scale) // 100)
            pygame.draw.line(screen, GREEN, (x1, y1), (x2, y2), int(2 * scale))
    if ALLOW_SEXUAL and female_ratio_history:
        if len(female_ratio_history) > max_entries:
            indices = np.linspace(0, len(female_ratio_history)-1, max_entries, dtype=int)
            draw_female = [female_ratio_history[i] for i in indices]
        else:
            draw_female = female_ratio_history
        for i in range(1, len(draw_female)):
            x1 = s_graph1_offset_x + (i-1) * int(500 * scale) // max_entries
            y1 = s_graph1_offset_y + int(300 * scale) - (draw_female[i-1] * 100 * int(300 * scale) // 100)
            x2 = s_graph1_offset_x + i * int(500 * scale) // max_entries
            y2 = s_graph1_offset_y + int(300 * scale) - (draw_female[i] * 100 * int(300 * scale) // 100)
            pygame.draw.line(screen, PINK, (x1, y1), (x2, y2), int(2 * scale))
    pop_label = font.render(f"Max Pop: {int(max_pop)}", True, BLACK)
    pop_label = pygame.transform.smoothscale_by(pop_label, scale)
    screen.blit(pop_label, (s_graph1_offset_x + int(350 * scale), s_graph1_offset_y))
    graph1_time = (time.perf_counter_ns() - graph1_start) / 1000
    
    # Draw offspring count, male investment, and lady's choice histograms (graph2)
    graph2_start = time.perf_counter_ns()
    pygame.draw.rect(screen, BLACK, (s_graph2_offset_x, s_graph2_offset_y, int(500 * scale), int(300 * scale)), 1)
    label_text = "Offspring"
    if ALLOW_SEXUAL:
        label_text += " / M.investm / F.choice"
    hist_label = font.render(label_text, True, BLACK)
    hist_label = pygame.transform.smoothscale_by(hist_label, scale)
    screen.blit(hist_label, (s_graph2_offset_x, s_graph2_offset_y - int(30 * scale)))
    
    # Lady's choice average histogram (if ALLOW_SEXUAL, 4 bins as columns, draw first)
    if ALLOW_SEXUAL:
        ladys_choice_data = [agent.ladys_choice for agent in agents if hasattr(agent, 'ladys_choice')]
        if ladys_choice_data:
            avg_ladys_choice = np.mean(ladys_choice_data, axis=0)
            if len(avg_ladys_choice) >= 4:  # Ensure at least 4 values
                bar_width = int(500 * scale) // 4
                for i in range(4):
                    x = s_graph2_offset_x + i * bar_width
                    height = int(avg_ladys_choice[i] * 300 * scale)  # Assume 0-1 range
                    y = s_graph2_offset_y + int(300 * scale) - height
                    pygame.draw.rect(screen, LIGHT_PINK, (x, y, bar_width, height))
    
    # Offspring count histogram (omit zero-offspring bin)
    offspring_counts = [len(agent.offspring) for agent in agents if len(agent.offspring) > 0]
    max_bin = 1
    offspring_points = []
    if offspring_counts:
        max_offspring = max(offspring_counts)
        offspring_bins, offspring_edges = np.histogram(offspring_counts, bins=HISTO_BINS, range=(1, max_offspring+1))
        max_bin = max(offspring_bins) or 1
        for i in range(len(offspring_bins)):
            x = s_graph2_offset_x + (offspring_edges[i] / max_offspring) * int(500 * scale)
            y = s_graph2_offset_y + int(300 * scale) - (offspring_bins[i] * int(300 * scale) // max_bin)
            offspring_points.append((x, y))
        for i in range(1, len(offspring_points)):
            pygame.draw.line(screen, YELLOW, offspring_points[i-1], offspring_points[i], int(2 * scale))
    
    # Male investment histogram (if ALLOW_SEXUAL)
    if ALLOW_SEXUAL:
        male_investments = [agent.male_investment for agent in agents if hasattr(agent, 'male_investment')]
        if male_investments:
            max_investment = max(male_investments)
            investment_bins, investment_edges = np.histogram(male_investments, bins=HISTO_BINS, range=(0, max_investment+1))
            max_bin = max(max_bin, max(investment_bins)) or 1
            investment_points = []
            for i in range(len(investment_bins)):
                x = s_graph2_offset_x + (investment_edges[i] / max_investment) * int(500 * scale)
                y = s_graph2_offset_y + int(300 * scale) - (investment_bins[i] * int(300 * scale) // max_bin)
                investment_points.append((x, y))
            for i in range(1, len(investment_points)):
                pygame.draw.line(screen, RED, investment_points[i-1], investment_points[i], int(2 * scale))
    graph2_time = (time.perf_counter_ns() - graph2_start) / 1000
    
    # Draw age, energy, and score histograms (graph3)
    graph3_start = time.perf_counter_ns()
    hist_label = font.render("Energy (R) / Score (G) / Age (B)", True, BLACK)
    hist_label = pygame.transform.smoothscale_by(hist_label, scale)
    screen.blit(hist_label, (s_graph3_offset_x, s_graph3_offset_y - int(30 * scale)))
    pygame.draw.rect(screen, BLACK, (s_graph3_offset_x, s_graph3_offset_y, int(500 * scale), int(300 * scale)), 1)
    ages = [turn - agent.born for agent in agents]
    energy_scores = [agent.energy for agent in agents]
    total_scores = [total_score[agent.id] for agent in agents]
    if ages and energy_scores and total_scores:
        max_age = max(ages) if ages else 1
        max_age += 1
        min_energy = min(energy_scores) if energy_scores else 0
        max_energy = max(energy_scores) if energy_scores else 1
        energy_range = max_energy - min_energy if max_energy != min_energy else 1
        min_score = min(total_scores) if total_scores else 0
        max_score = max(total_scores) if total_scores else 1
        score_range = max_score - min_score if max_score != min_score else 1
        age_bins, age_edges = np.histogram(ages, bins=HISTO_BINS, range=(0, max_age))
        energy_bins, energy_edges = np.histogram(energy_scores, bins=HISTO_BINS, range=(min_energy, max_energy))
        score_bins, score_edges = np.histogram(total_scores, bins=HISTO_BINS, range=(min_score, max_score))
        max_bin = max(max(age_bins), max(energy_bins), max(score_bins)) or 1
        # Plot age histogram (blue)
        age_points = []
        for i in range(len(age_bins)):
            x = s_graph3_offset_x + (age_edges[i] / max_age) * int(500 * scale)
            y = s_graph3_offset_y + int(300 * scale) - (age_bins[i] * int(300 * scale) // max_bin)
            age_points.append((x, y))
        for i in range(1, len(age_points)):
            pygame.draw.line(screen, BLUE, age_points[i-1], age_points[i], int(2 * scale))
        # Plot energy histogram (red)
        energy_points = []
        for i in range(len(energy_bins)):
            x = s_graph3_offset_x + ((energy_edges[i] - min_energy) / energy_range) * int(500 * scale)
            y = s_graph3_offset_y + int(300 * scale) - (energy_bins[i] * int(300 * scale) // max_bin)
            energy_points.append((x, y))
        for i in range(1, len(energy_points)):
            pygame.draw.line(screen, RED, energy_points[i-1], energy_points[i], int(2 * scale))
        # Plot score histogram (green)
        score_points = []
        for i in range(len(score_bins)):
            x = s_graph3_offset_x + ((score_edges[i] - min_score) / score_range) * int(500 * scale)
            y = s_graph3_offset_y + int(300 * scale) - (score_bins[i] * int(300 * scale) // max_bin)
            score_points.append((x, y))
        for i in range(1, len(score_points)):
            pygame.draw.line(screen, GREEN, score_points[i-1], score_points[i], int(2 * scale))
        # Draw x-axis labels for energy
        for i in range(6):
            energy_label = min_energy + i * (max_energy - min_energy) / 5
            label = font.render(f"{energy_label:.0f}", True, BLACK)
            label = pygame.transform.smoothscale_by(label, scale)
            screen.blit(label, (s_graph3_offset_x + i * int(100 * scale) - int(10 * scale), s_graph3_offset_y + int(310 * scale)))
        # Draw max age label
        age_label = font.render(f"Max Age: {int(max_age)}", True, BLACK)
        age_label = pygame.transform.smoothscale_by(age_label, scale)
        screen.blit(age_label, (s_graph3_offset_x + int(400 * scale), s_graph3_offset_y + int(340 * scale)))
    graph3_time = (time.perf_counter_ns() - graph3_start) / 1000
    
    # Draw xenophobia histogram and identity scatterplot (graph4)
    graph4_start = time.perf_counter_ns()
    pygame.draw.rect(screen, BLACK, (s_graph4_offset_x, s_graph4_offset_y, int(500 * scale), int(300 * scale)), 1)
    hist_label = font.render("Xenophobia " + (" / Identity(A,B)" if SCATTERPLOT_ENABLED else ""), True, BLACK)
    hist_label = pygame.transform.smoothscale_by(hist_label, scale)
    screen.blit(hist_label, (s_graph4_offset_x, s_graph4_offset_y - int(30 * scale)))
    xenophobia = [agent.ingroup_preference for agent in agents]
    if SCATTERPLOT_ENABLED:
        scatter_start = time.perf_counter_ns()
        # Orthonormalize TRIBE_MAP["A"] and TRIBE_MAP["B"]
        u1 = np.array(TRIBE_MAP["A"])
        u2 = np.array(TRIBE_MAP["B"])
        if np.all(u1 == 0):
            e1 = u2 / np.linalg.norm(u2)
            # Choose orthogonal vector, e.g., swap non-zero component
            idx = np.argmax(np.abs(e1))
            e2 = np.zeros_like(e1)
            e2[(idx + 1) % IDENTITY_DIMENSIONS] = 1
            e2 = e2 - np.dot(e2, e1) * e1
            e2 = e2 / np.linalg.norm(e2)
        else:
            e1 = u1 / np.linalg.norm(u1)
            u2_prime = u2 - np.dot(u2, u1) / np.dot(u1, u1) * u1
            e2 = u2_prime / np.linalg.norm(u2_prime)
        
        # Project agent.identity onto plane
        identity_points = []
        for agent in agents:
            v = np.array(agent.identity)
            x = np.dot(v, e1)
            y = np.dot(v, e2)
            if -10 <= x <= 10 and -10 <= y <= 10:  # Retain filtering
                identity_points.append((x, y))
        
        scatter_gather_time = (time.perf_counter_ns() - scatter_start) / 1000
        scatter_draw_start = time.perf_counter_ns()
        if identity_points:
            x_coords, y_coords = zip(*identity_points)
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_range = x_max - x_min if x_max != x_min else 1
            y_range = y_max - y_min if y_max != y_min else 1
            for x, y in identity_points:
                px = s_graph4_offset_x + ((x - x_min) / x_range) * int(500 * scale)
                py = s_graph4_offset_y + int(300 * scale) - ((y - y_min) / y_range) * int(300 * scale)
                dot_rect = pygame.Rect(px - int(1.5 * scale), py - int(1.5 * scale), int(3 * scale), int(3 * scale))
                pygame.draw.rect(screen, BLUE, dot_rect)
        scatter_draw_time = (time.perf_counter_ns() - scatter_draw_start) / 1000
    else:
        scatter_gather_time = 0
        scatter_draw_time = 0
    if xenophobia:
        xe_bins, xe_edges = np.histogram(xenophobia, bins=HISTO_BINS, range=(0, 1))
        max_bin = max(xe_bins) or 1
        xe_points = []
        for i in range(len(xe_bins)):
            x = s_graph4_offset_x + xe_edges[i] * int(500 * scale)
            y = s_graph4_offset_y + int(300 * scale) - (xe_bins[i] * int(300 * scale) // max_bin)
            xe_points.append((x, y))
        for i in range(1, len(xe_points)):
            pygame.draw.line(screen, GREEN, xe_points[i-1], xe_points[i], int(2 * scale))
    graph4_time = (time.perf_counter_ns() - graph4_start) / 1000
    
    # Draw numeric data
    data_start = time.perf_counter_ns()
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
        rendered = pygame.transform.smoothscale_by(rendered, scale)
        screen.blit(rendered, (s_data_offset_x, s_data_offset_y + i * int(30 * scale)))
    data_time = (time.perf_counter_ns() - data_start) / 1000
    
    # Draw pause/exit button
    button_start = time.perf_counter_ns()
    button_rect = pygame.Rect(s_button_offset_x, s_button_offset_y, int(100 * scale), int(40 * scale))
    button_color = GRAY if paused and not simulation_ended else YELLOW
    pygame.draw.rect(screen, button_color, button_rect)
    button_text = "Exit" if simulation_ended else "Pause" if not paused else "Resume"
    text_key = f"button_{button_text}_{int(24 * scale)}"
    if text_key not in text_cache:
        text = font.render(button_text, True, BLACK)
        text = pygame.transform.smoothscale_by(text, scale)
        text_cache[text_key] = text
    text = text_cache[text_key]
    text_rect = text.get_rect(center=button_rect.center)
    screen.blit(text, text_rect)
    button_time = (time.perf_counter_ns() - button_start) / 1000
    
    # Display flip
    flip_start = time.perf_counter_ns()
    pygame.display.flip()
    flip_time = (time.perf_counter_ns() - flip_start) / 1000
    
    # Print timing breakdown in ms
    total_time_ms = (time.perf_counter_ns() - start_time) / 1000000
    canvas_time_ms = canvas_time / 1000
    grid_time_ms = grid_time / 1000
    graph1_time_ms = graph1_time / 1000
    graph2_time_ms = graph2_time / 1000
    graph3_time_ms = graph3_time / 1000
    graph4_time_ms = graph4_time / 1000
    scatter_gather_time_ms = scatter_gather_time / 1000
    scatter_draw_time_ms = scatter_draw_time / 1000
    data_time_ms = data_time / 1000
    button_time_ms = button_time / 1000
    flip_time_ms = flip_time / 1000
    if (turn % PRINT_INTERVAL == 0 or turn < DEBUG_TURNS):
        print(f"Draw time: {total_time_ms:.1f} ms (Canvas: {canvas_time_ms:.1f}, Grid: {grid_time_ms:.1f}, Graph1: {graph1_time_ms:.1f}, Graph2: {graph2_time_ms:.1f}, Graph3: {graph3_time_ms:.1f}, Graph4: {graph4_time_ms:.1f}, Scatter Gather: {scatter_gather_time_ms:.1f}, Scatter Draw: {scatter_draw_time_ms:.1f}, Data: {data_time_ms:.1f}, Button: {button_time_ms:.1f}, Flip: {flip_time_ms:.1f})")

def handle_button_click(pos, paused, simulation_ended):
    window_width, window_height = pygame.display.get_surface().get_size()
    window_aspect = window_width / window_height
    if window_aspect > ASPECT_RATIO:
        canvas_width = int(window_height * ASPECT_RATIO)
        canvas_height = window_height
    else:
        canvas_width = window_width
        canvas_height = int(window_width / ASPECT_RATIO)
    canvas_width = max(MIN_WIDTH, min(MAX_WIDTH, canvas_width))
    scale = canvas_width / BASE_WIDTH
    canvas_x = (window_width - canvas_width) // 2
    canvas_y = (window_height - canvas_height) // 2
    s_button_offset_x = int(BUTTON_OFFSET_X * scale) + canvas_x
    s_button_offset_y = int(BUTTON_OFFSET_Y * scale) + canvas_y
    button_rect = pygame.Rect(s_button_offset_x, s_button_offset_y, int(100 * scale), int(40 * scale))
    if button_rect.collidepoint(pos):
        if simulation_ended:
            return True, paused  # Exit
        else:
            return False, not paused  # Toggle pause
    return False, paused

###############################################################################################################
# main loop

def main(control_queue, param_queue):
    global agents, next_id, history_dict, id_to_agent, population_history, cc_percentage_history, turn, paused, timestamp, NUM_TURNS, FPS
    global screen, font
    
    last_logged_turn = 0
    # Wait for initial parameters
    params = param_queue.get()  # Blocking
    print(f"Received params: {params}")
    # Update constants initially
    update_constants(params)

    # Initialize founding agents
    print (f"received ingroup_pref1 {params[f"ingroup_pref1"]}, ingroup_pref2  {params[f"ingroup_pref2"]}")
    founders = [(i, params[f"founder{i}"], params[f"strategy{i}"], params[f"tribe{i}"]) for i in range(1, 6)]
    active_founders = [(i, strategy, tribe) for i, active, strategy, tribe in founders if active]
    if not active_founders:
        print("Warning: No founders selected, defaulting to founder1")
        active_founders = [(1, params["strategy1"], params["tribe1"])]
    if params["allow_sexual"]:
        print(f"Spawning {len(active_founders)} breeding pairs (sexual reproduction)")
        for idx, (founder_num, strategy, tribe) in enumerate(active_founders):
            posx, posy = FOUNDER_POSITIONS[idx % len(FOUNDER_POSITIONS)]
            # Male
            append_agent(
                posx=posx, posy=posy, id=next_id, born=0, sex=0,
                identity=TRIBE_MAP[tribe], ingroup_preference=params[f"ingroup_pref{founder_num}"],
                strategy=strategy, sexual_selection="fascist",
                agents=agents
            )
            next_id += 1
            # Female
            append_agent(
                posx=posx, posy=posy, id=next_id, born=0, sex=1,
                identity=TRIBE_MAP[tribe], ingroup_preference=params[f"ingroup_pref{founder_num}"],
                strategy=strategy, sexual_selection="fascist",
                agents=agents
            )
            next_id += 1
    else:
        print(f"Spawning {len(active_founders)} asexual agents")
        for idx, (founder_num, strategy, tribe) in enumerate(active_founders):
            posx, posy = FOUNDER_POSITIONS[idx % len(FOUNDER_POSITIONS)]
            # Asexual agent
            append_agent(
                posx=posx, posy=posy, id=next_id, born=0,
                identity=TRIBE_MAP[tribe], ingroup_preference=params[f"ingroup_pref{founder_num}"],
                strategy=strategy, sexual_selection="balanced",
                agents=agents
            )
            next_id += 1
    print(f"Initialized {len(agents)} agents")
    print(f"Before setting turn: {turn if 'turn' in globals() else 'undefined'}")

    # initialize Pygame display
    pygame.init()
    pygame_width = int(screen_width * 0.65) 
    pygame_height = int(screen_height * 0.7) 
#    os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0" # to force window placement   
    pygame_screen = pygame.display.set_mode((pygame_width, pygame_height), pygame.RESIZABLE)
    pygame.display.set_caption("Simulation State")
    #font = pygame.font.SysFont('arial', 24)
    font = pygame.font.SysFont('monospace', 24)  # use fixed-width font for now
    text_cache = {}  # Shared cache
    screen = pygame_screen
    pygame_clock = pygame.time.Clock()

    while turn < NUM_TURNS:
    
        try:
            new_params = param_queue.get_nowait()
#            print(f"Updated params: {new_params}")
            update_constants(new_params)
        except queue.Empty:
            pass
                    
        timestamp = time.time()
        start_time = time.perf_counter()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                should_exit, paused = handle_button_click(event.pos, paused, False)
                if should_exit:
                    return        
        if paused:
            draw_visualization(screen, agents, cc_games, dd_games, interactions, population_history, cc_percentage_history, female_ratio_history, total_score, population, font, turn, 0, paused, text_cache)
            time.sleep(1.0 / FPS)
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
        female_ratio_history.append(0 if len(agents) == 0 else female_pop/len(agents))

    # Update visualization every DRAW_INTERVAL-th turn
        if turn % DRAW_INTERVAL == 0:
            draw_visualization(screen, agents, cc_games, dd_games, interactions, population_history, cc_percentage_history, female_ratio_history, total_score, population, font, turn,  (time.time()-timestamp), paused, text_cache)

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

        elapsed = time.perf_counter() - start_time
        frame_time = 1.0 / FPS
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

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
        
        draw_visualization(screen, agents, cc_games, dd_games, interactions, population_history, cc_percentage_history, female_ratio_history, total_score, population, font, turn,  0, paused, text_cache, True)
        time.sleep(1.0 / FPS)

#######################################################################
#Tkinter gui (start screen)
def create_start_screen():
    global screen_width, screen_height
    # Initialize Tkinter window
    root = tk.Tk()
    root.title("The Harshest Mistress - Start Screen")

    # Get screen resolution and set dynamic window size (~25% of screen width, 75% height)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = int(screen_width * 0.25)
    window_height = int(screen_height * 0.75)
    root.geometry(f"{window_width}x{window_height}")
    root.resizable(True, True)
    root.minsize(300, 500)
    root.maxsize(int(screen_width * 0.5), int(screen_height ))

    # Parameters dictionary
    params = {
        "num_turns": tk.StringVar(value="500"),
        "interaction_depth": tk.StringVar(value="3"),
        "base_energy": tk.IntVar(value=100),
        "consumption": tk.IntVar(value=25),
        "migration_rate": tk.DoubleVar(value=0.05),
        "migration_cost": tk.IntVar(value=5),
        "aging_penalty": tk.DoubleVar(value=0.4),
        "mutation_rate": tk.DoubleVar(value=0.1),
        "mutation_rate_tribal": tk.DoubleVar(value=0.05),
        "mutation_rate_sexual": tk.DoubleVar(value=0.05),
        "log_to_file": tk.BooleanVar(value=False),
        "logfile": tk.StringVar(value=logfile_path),
        "log_turns": tk.StringVar(value="100"),
        "allow_asexual": tk.BooleanVar(value=False),
        "allow_sexual": tk.BooleanVar(value=True),
        "founder1": tk.BooleanVar(value=True),
        "strategy1": tk.StringVar(value="TFT"),
        "tribe1": tk.StringVar(value="A"),
        "ingroup_pref1": tk.DoubleVar(value=0.5),
        "founder2": tk.BooleanVar(value=False),
        "strategy2": tk.StringVar(value="TFT"),
        "tribe2": tk.StringVar(value="B"),
        "ingroup_pref2": tk.DoubleVar(value=0.5),
        "founder3": tk.BooleanVar(value=False),
        "strategy3": tk.StringVar(value="TFT"),
        "tribe3": tk.StringVar(value="C"),
        "ingroup_pref3": tk.DoubleVar(value=0.5),
        "founder4": tk.BooleanVar(value=False),
        "strategy4": tk.StringVar(value="TFT"),
        "tribe4": tk.StringVar(value="D"),
        "ingroup_pref4": tk.DoubleVar(value=0.5),
        "founder5": tk.BooleanVar(value=False),
        "strategy5": tk.StringVar(value="TFT"),
        "tribe5": tk.StringVar(value="E"),
        "ingroup_pref5": tk.DoubleVar(value=0.5),
    }

    # Font scaling setup
    SCALE_EXP = 0.8
    base_font_size = 12
    min_font_size = 8
    max_font_size = 30
    title_font = tkfont.Font(family="Arial", size=base_font_size, weight="bold")
    label_font = tkfont.Font(family="Arial", size=base_font_size)
    small_font = tkfont.Font(family="Arial", size=(base_font_size-2))

    # Display variables for ingroup_pref sliders
    ingroup_display = {}
    for i in range(1, 6):
        ingroup_display[i] = tk.StringVar()
        def update_display(i=i):
            ingroup_display[i].set(f"{params[f'ingroup_pref{i}'].get():.2f}")
        params[f"ingroup_pref{i}"].trace("w", lambda *args, i=i: update_display(i))
        update_display(i)  # Initialize display

    # Parameter change callback
    def on_param_change(*args):
        param_values = {key: var.get() for key, var in params.items()}
        param_queue.put(param_values)

    # Trace all params except checkboxes
    for key, var in params.items():
        if key not in ["allow_asexual", "allow_sexual", "log_to_file", "founder1", "founder2", "founder3", "founder4", "founder5"]:
            var.trace("w", on_param_change)

    sim_process = None
    control_queue = Queue()
    param_queue = Queue()

    def update_fonts(event=None):
        new_width = root.winfo_width()
        new_height = root.winfo_height()
        scale_factor = min((new_width / window_width) ** SCALE_EXP, (new_height / window_height) ** SCALE_EXP)
        new_font_size = int(base_font_size * scale_factor)
        new_font_size = max(min_font_size, min(new_font_size, max_font_size))
        title_font.configure(size=new_font_size + 2)
        label_font.configure(size=new_font_size)
        small_font.configure(size=max(new_font_size - 2, min_font_size))

    def update_scroll_region(event=None):
        main_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

    def select_logfile():
        filepath = filedialog.asksaveasfilename(defaultextension=".log", filetypes=[("log files", "*.log"), ("txt files", "*.txt"), ("All files", "*.*")])
        if filepath:
            params["logfile"].set(filepath)

    root.bind("<Configure>", update_fonts)

    def update_canvas_alignment(event=None):
        canvas.coords(canvas.create_window(0, 0, window=main_frame, anchor="n"), event.width / 2, 0)

    canvas = tk.Canvas(root, width=window_width, height=window_height)
    scrollbar_y = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollbar_x = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)
    canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
    scrollbar_y.pack(side="right", fill="y")
    scrollbar_x.pack(side="bottom", fill="x")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.bind("<Configure>", update_canvas_alignment)

    #tk is now drawing "frames", i.e. main_frame, and within it rules_frame, founders_frame, reproduction_frame, logging_frame, button_frame, some of them with subframes to achieve a html-table like structure of gui elements
    main_frame = tk.Frame(canvas)
    canvas.create_window((window_width/2, 0), window=main_frame, anchor="n")

    main_frame.bind("<Configure>", update_scroll_region)

    title_label = tk.Label(main_frame, text="The Harshest Mistress", font=title_font)
    title_label.pack(pady=5)
    subtitle_label = tk.Label(main_frame, text="Prisoner's Dilemma evolved", font=label_font)
    subtitle_label.pack(pady=5)

    rules_col='#d0d0f0'
    rules_frame = tk.Frame(main_frame, bg=rules_col)
    rules_frame.pack(pady=10)
    tk.Label(rules_frame, text="game rules", font=label_font, bg=rules_col).pack()
    entry_row = tk.Frame(rules_frame, bg=rules_col)
    entry_row.pack(fill="x", pady=2)
    tk.Label(entry_row, text="turns:", font=label_font, bg=rules_col).grid(row=0, column=0, sticky="e", padx=5)
    turns_subframe = tk.Frame(entry_row, bg=rules_col)
    turns_subframe.grid(row=0, column=1, sticky="w", padx=5)
    turns_entry = tk.Entry(turns_subframe, textvariable=params["num_turns"], width=5)
    turns_entry.pack(side="left")
    entry_row.grid_columnconfigure(1, weight=1)
    tk.Label(entry_row, text="depth:", font=label_font, bg=rules_col).grid(row=0, column=2, sticky="e", padx=5)
    depth_subframe = tk.Frame(entry_row, bg=rules_col)
    depth_subframe.grid(row=0, column=3, sticky="w", padx=5)
    depth_entry = tk.Entry(depth_subframe, textvariable=params["interaction_depth"], width=3)
    depth_entry.pack(side="left")
    entry_row.grid_columnconfigure(3, weight=1)

    slider_row = tk.Frame(rules_frame, bg=rules_col)
    slider_row.pack(fill="x", pady=2)
    tk.Label(slider_row, text="energy", font=small_font, bg=rules_col).grid(row=0, column=0, padx=5)
    energy_slider = tk.Scale(slider_row, bg=rules_col, from_=10, to=200, resolution=5, orient=tk.VERTICAL, variable=params["base_energy"], length=int(100 * min((window_width/422) ** SCALE_EXP, 1.5)))
    energy_slider.grid(row=1, column=0, padx=5)
    tk.Label(slider_row, text="consumpt", font=small_font, bg=rules_col).grid(row=0, column=1, padx=5)
    consumption_slider = tk.Scale(slider_row, bg=rules_col, from_=1, to=50, resolution=1, orient=tk.VERTICAL, variable=params["consumption"], length=int(100 * min((window_width/422) ** SCALE_EXP, 1.5)))
    consumption_slider.grid(row=1, column=1, padx=5)
    tk.Label(slider_row, text="migration", font=small_font, bg=rules_col).grid(row=0, column=2, padx=5)
    migrate_slider = tk.Scale(slider_row, bg=rules_col, from_=0.0, to=0.5, resolution=0.05, orient=tk.VERTICAL, variable=params["migration_rate"], length=int(100 * min((window_width/422) ** SCALE_EXP, 1.5)))
    migrate_slider.grid(row=1, column=2, padx=5)
    tk.Label(slider_row, text="migr. cost", font=small_font, bg=rules_col).grid(row=0, column=3, padx=5)
    migcost_slider = tk.Scale(slider_row, bg=rules_col, from_=0, to=25, resolution=1, orient=tk.VERTICAL, variable=params["migration_cost"], length=int(100 * min((window_width/422) ** SCALE_EXP, 1.5)))
    migcost_slider.grid(row=1, column=3, padx=5)
    tk.Label(slider_row, text="aging", font=small_font, bg=rules_col).grid(row=0, column=4, padx=5)
    aging_slider = tk.Scale(slider_row, bg=rules_col, from_=0.0, to=1.0, resolution=0.1, orient=tk.VERTICAL, variable=params["aging_penalty"], length=int(100 * min((window_width/422) ** SCALE_EXP, 1.5)))
    aging_slider.grid(row=1, column=4, padx=5)

    founders_col='#d0f0d0'
    founders_frame = tk.Frame(main_frame, bg=founders_col)
    founders_frame.pack(pady=12)
    tk.Label(founders_frame, text="founder population", font=label_font, bg=founders_col).pack()
    tk.Label(founders_frame, text="(set strategy, tribe, ingroup preference)", font=small_font, bg=founders_col).pack()
    column_row = tk.Frame(founders_frame, bg=founders_col)
    column_row.pack(fill="x", pady=2)
    founder_labels = ["top-left", "lower-right", "lower-left", "top-right", "center"]
    for i in range(1, 6):
        col = i - 1
        tk.Label(column_row, text=founder_labels[col], font=small_font, bg=founders_col).grid(row=0, column=col, sticky="w", padx=5)
        tk.Checkbutton(column_row, bg=founders_col, variable=params[f"founder{i}"]).grid(row=1, column=col, sticky="w", padx=5)
        tk.OptionMenu(column_row, params[f"strategy{i}"], "TFT", "TFTT", "AC", "AD", "GT", "random").grid(row=2, column=col, sticky="w", padx=5)
        tk.OptionMenu(column_row, params[f"tribe{i}"], "A", "B", "C", "D", "E").grid(row=3, column=col, sticky="w", padx=5)
        ttk.Scale(column_row, from_=0.0, to=1.0, orient=tk.VERTICAL, variable=params[f"ingroup_pref{i}"], length=int(50 * min((window_width/422) ** SCALE_EXP, 1.5))).grid(row=4, column=col, padx=5)
        tk.Label(column_row, textvariable=ingroup_display[i], font=small_font, width=4, bg=founders_col).grid(row=5, column=col, padx=5)

    reproduction_col='#f0d0d0'
    reproduction_frame = tk.Frame(main_frame, bg=reproduction_col)
    reproduction_frame.pack(pady=10)
    tk.Label(reproduction_frame, text="reproduction", font=label_font, bg=reproduction_col).pack()
    checkbox_row = tk.Frame(reproduction_frame, bg=reproduction_col)
    checkbox_row.pack(fill="x", pady=2)

    def on_checkbox_change(*args):
        if not params["allow_asexual"].get() and not params["allow_sexual"].get():
            if args[0] == "allow_asexual":
                params["allow_sexual"].set(True)
            else:
                params["allow_asexual"].set(True)

    params["allow_asexual"].trace("w", lambda *args: on_checkbox_change("allow_asexual"))
    params["allow_sexual"].trace("w", lambda *args: on_checkbox_change("allow_sexual"))

    tk.Label(checkbox_row, text="asexual:", font=small_font, bg=reproduction_col).grid(row=0, column=0, sticky="e", padx=5)
    asexual_check = tk.Checkbutton(checkbox_row, bg=reproduction_col, variable=params["allow_asexual"])
    asexual_check.grid(row=0, column=1, sticky="w", padx=5)
    tk.Label(checkbox_row, text="sexual:", font=small_font, bg=reproduction_col).grid(row=0, column=2, sticky="e", padx=5)
    sexual_check = tk.Checkbutton(checkbox_row, bg=reproduction_col, variable=params["allow_sexual"])
    sexual_check.grid(row=0, column=3, sticky="w", padx=5)

    tk.Label(reproduction_frame, text="mutation rates", font=small_font, bg=reproduction_col).pack()
    slider_row = tk.Frame(reproduction_frame, bg=reproduction_col)
    slider_row.pack(fill="x", pady=2)
    tk.Label(slider_row, text="individual", font=small_font, bg=reproduction_col).grid(row=0, column=0, padx=5)
    mutation_slider = tk.Scale(slider_row, bg=reproduction_col, from_=0.0, to=0.5, resolution=0.01, orient=tk.VERTICAL, variable=params["mutation_rate"], length=int(100 * min((window_width/422) ** SCALE_EXP, 1.5)))
    mutation_slider.grid(row=1, column=0, padx=5)
    tk.Label(slider_row, text="tribal", font=small_font, bg=reproduction_col).grid(row=0, column=1, padx=5)
    tribal_slider = tk.Scale(slider_row, bg=reproduction_col, from_=0.0, to=0.5, resolution=0.01, orient=tk.VERTICAL, variable=params["mutation_rate_tribal"], length=int(100 * min((window_width/422) ** SCALE_EXP, 1.5)))
    tribal_slider.grid(row=1, column=1, padx=5)
    tk.Label(slider_row, text="sexual", font=small_font, bg=reproduction_col).grid(row=0, column=2, padx=5)
    sexual_slider = tk.Scale(slider_row, bg=reproduction_col, from_=0.0, to=0.5, resolution=0.01, orient=tk.VERTICAL, variable=params["mutation_rate_sexual"], length=int(100 * min((window_width/422) ** SCALE_EXP, 1.5)))
    sexual_slider.grid(row=1, column=2, padx=5)

    logging_frame = tk.Frame(main_frame)
    logging_frame.pack(pady=10)
    tk.Label(logging_frame, text="logging", font=label_font).pack()
    logcheckbox_row = tk.Frame(logging_frame)
    logcheckbox_row.pack(fill="x", pady=2)
    tk.Label(logcheckbox_row, text="log to file", font=label_font).grid(row=0, column=0, sticky="e", padx=5)
    log_check = ttk.Checkbutton(logcheckbox_row, variable=params["log_to_file"])
    log_check.grid(row=0, column=1, sticky="w", padx=5)

    logfile_frame = tk.Frame(logging_frame)
    logfile_frame.pack(fill=tk.X, pady=2)
    logfile_entry = tk.Entry(logfile_frame, textvariable=params["logfile"], width=20)
    logfile_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    tk.Button(logfile_frame, text="Browse", command=select_logfile).pack(side=tk.LEFT, padx=5)

    logturns_row = tk.Frame(logging_frame)
    logturns_row.pack(fill="x", pady=2)
    tk.Label(logturns_row, text="logging turns:", font=label_font).grid(row=0, column=0, sticky="e", padx=5)
    logturns_subframe = tk.Frame(logturns_row)
    logturns_subframe.grid(row=0, column=1, sticky="w", padx=5)
    logturns_entry = tk.Entry(logturns_subframe, textvariable=params["log_turns"], width=5)
    logturns_entry.pack(side="left")
    logturns_row.grid_columnconfigure(1, weight=1)

    button_frame = tk.Frame(main_frame)
    button_frame.pack(pady=10)
    start_button = tk.Button(button_frame, text="Start", width=10)
    start_button.pack(side=tk.LEFT, padx=5)

    def toggle_simulation():
        nonlocal sim_process
        if sim_process is None:
            param_values = {key: var.get() for key, var in params.items()}
            print(f"Sending initial param_values: {param_values}")
            # Clear param_queue and add dummy write
            while not param_queue.empty():
                try:
                    param_queue.get_nowait()
                except Queue.Empty:
                    break
            param_queue.put({"dummy": True})
            param_queue.get()  # Remove dummy

            param_queue.put(param_values)
            time.sleep(0.1)  # 100ms delay to ensure queue write
            sim_process = Process(target=main, args=(control_queue, param_queue))
            sim_process.start()
            start_button.config(text="Stop")
        else:
            control_queue.put("stop")
            if sim_process:
                sim_process.terminate()
                sim_process.join()
                sim_process = None
            start_button.config(text="Start")

    start_button.config(command=toggle_simulation)

    def exit_app():
        nonlocal sim_process
        if sim_process:
            control_queue.put("stop")
            sim_process.terminate()
            sim_process.join()
            sim_process = None
        root.destroy()
        sys.exit(0)

    tk.Button(button_frame, text="Exit", command=exit_app, width=10).pack(side=tk.LEFT, padx=5)
    main_frame.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))

###############################################################
    # Run Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    try:
        create_start_screen()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
