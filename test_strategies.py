import numpy as np
from sample_strategies import sample_strategies
from itertools import product

#Auxiliary script to test the hand-crafted Prisoner's Dilemma strategies encoded in sample_strategies.py


test_names = ["evolved"] # Pick strategies to test
#we loop over all "recent" (last three turns) combinations below. The following allows us to set a custom value for the "remote memory" (averages over actions more than three turns ago) parameter.
test_remote_memories = [(0.0,0.0),(1.0,1.0),(-1.0,1.0),(1.0,-1.0)]
turns_total = 100  #this is just to simulate interaction history length in order to be able to update remote weights (own_avg, opp_avg) with applicable recent history

# Agent class
class Agent:
    def __init__(self, cooperation_bias, remote_weights, recent_weights, contrition, bigotry):
        self.cooperation_bias = np.float16(cooperation_bias)
        self.remote_weights = np.array(remote_weights, dtype=np.float16)
        self.recent_weights = np.array(recent_weights, dtype=np.float16)
        self.contrition = np.float16(contrition)
        self.bigotry = np.float16(bigotry)

########
#decision algorithm tested
    def decide(self, input_vector):
        if input_vector[2] == 0: #empty history check, does not apply in this test scenario
            total_output = self.cooperation_bias
        else:
            scaled_vector = input_vector.copy()
            #apply contrition/bigotry:
            #contrition: weigh own defections more harshly
            for i in [0,2,3,4]:  # own_avg, own1/2/3
                if scaled_vector[i] == -1:
                    scaled_vector[i] *= self.contrition
                #CAUTION: inverting "own" values
                #scaled_vector[i] *= -1
            #bigotry: weigh opponent's defections more harshly
            for i in [1,5,6,7]:  # opp_avg, opp1/2/3
                if scaled_vector[i] == -1:
                    scaled_vector[i] *= self.bigotry
            #print (f"{input_vector} -> {scaled_vector}")
            remote_term = np.dot(np.abs(scaled_vector[:2]), np.dot(self.remote_weights, scaled_vector[:2]))
            recent_term = np.dot(np.abs(scaled_vector[2:]), np.dot(self.recent_weights, scaled_vector[2:]))
            total_output = remote_term + recent_term + self.cooperation_bias
        return total_output
########

for (own_avg,opp_avg) in test_remote_memories:
    for name in test_names:
        if name not in sample_strategies:
            print(f"Error: Strategy {name} not found")
            continue
        print(f"\nStrategy: {name}, own_avg={own_avg:.2f}, opp_avg={opp_avg:.2f}")
        strategy = sample_strategies[name]
        agent = Agent(
            cooperation_bias=strategy["cooperation_bias"],
            remote_weights=strategy["remote_weights"],
            recent_weights=strategy["recent_weights"],
            contrition=strategy["contrition"],
            bigotry=strategy["bigotry"]
        )
    
        # Generate combinations
        actions = [1, -1]
        self_combinations = list(product(actions, repeat=3))  # self_last1/2/3
        opp_combinations = list(product(actions, repeat=3))   # opp_last1/2/3
        opp_headers = ["".join("C" if x == 1 else "D" for x in opp) for opp in opp_combinations]
        
        # Compute and print table
        table=[]
        for self1, self2, self3 in self_combinations:
            self_label = "".join("C" if x == 1 else "D" for x in (self1, self2, self3))
            row = []
            for opp1, opp2, opp3 in opp_combinations:
                current_own_avg=own_avg/(1+((turns_total-3)/turns_total))+(self1+self2+self3)/turns_total
                current_opp_avg=opp_avg/(1+((turns_total-3)/turns_total))+(opp1+opp2+opp3)/turns_total
                input_vector = np.array([current_own_avg, current_opp_avg, self1, self2, self3, opp1, opp2, opp3], dtype=np.float16)
                
                total_output = agent.decide(input_vector)
                prob_cooperate = np.clip(0.5 + 0.1 * total_output, 0, 1)
                percent = int(prob_cooperate * 100)
                #print(f"input: {input_vector}, score: {total_output:.2f}, prob: {prob_cooperate:.2f}")                
                row.append(f"{percent:>3}")
            table.append(f"{self_label:>3} {' '.join(row)}\n")
        print (f"\n")
        print("    ", " ".join(f"{h:>3}" for h in opp_headers))
        print (f"{''.join(table)}")
        
