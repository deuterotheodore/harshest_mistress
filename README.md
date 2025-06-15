# "The Harshest Mistress"
Evolutionary simulation

Tested using: Python 3.12.3, requires: pygame

### General description

Agents live on a 10x10 grid and can interact with agents in neighboring cells (Moore neighborhood).
Each cell receives a fixed amount of energy per turn (EXTRACTION_BASE) and each agent has an energy value and a fixed metabolism cost (CONSUMPTION).
If agents' energy falls below zero, they die. Energy received by the cell is shared between agents in that cell, imposing a population cap.
Agents will age by gradual increase of their energy consumption as they become older (AGING_PENALTY).

Agents can reproduce if they reach a certain energy threshold. Asexual and sexual reproduction is available, controlled by the flags ASEXUAL_ALLOWED and SEXUAL_ALLOWED. In asexual reproduction, agent sex is irrelevant and they simply clone themselves into their own or a neighboring cell, applying mutations to their genome (Gaussian noise). Sexual reproduction is more complicated, discussed below.

Agents may migrate to neighboring cells with lower population. At present, the decision to migrate isn't in any way affected by genetic predisposition, there is just a fixed MIGRATION_RATE probability. Migration is mostly relevant during early game to populate the board and likely becomes irrelevant once the board is fully populated.
#Agents may already "raid" neigboring cells for energy and males may spawn offspring into neighboring cells, even without migration. Maybe  add more complex migration mechanic in the future, genetic parameter self.mobility, tailored to sexual reproduction (move away from overpopulated cells but stay in vicinity of populated cells for reproduction opportunities)

Agents interact by playing "Prisoner's Dilemma" (PD) with all agents in their neigborhood, competing for available energy. The number of PD interactions played each turn is INTERACTION_DEPTH. Each PD interaction leads to payout of energy proportional to the score achieved. A bonus for mutual cooperation and a penalty for mutual defection is applied (COOPERATION_BONUS, DEFECTION_PENALTY).

Agent behavior is controlled by their genome. The genome is divided into three strata or domains, "individualistic", "tribal" and "sexual". "Individualistic" behavior describes the strategy applied in Prisoner's Dilemma games. "Tribal" behavior applies modifiers on cooperation based on a metric of how closely interacting agents are related in "ethnic identity". "Sexual" behavior adds sex-specific modifiers to cooperation based on sex and affects mating behavior in sexual reproduction.

In order to run simulations focussed on either of these domains, mutation rates can be set separately, (MUTATION_RATE, MUTATION_RATE_IDENTITY, MUTATION_RATE_SEXUAL = 0.1). E.g. in order to just explore sexual selection, set MUTATION_RATE=0, MUTATION_RATE_IDENTITY=0 and e.g. MUTATION_RATE_SEXUAL=0.1. Initialize one or several breeding pairs with fixed strategies (such as "tit-for-tat") that now will remain constant over the whole simulation, so that only sexual selection will play out evolutionarily. And mutatis mutandis for the other domains, or enable all three domains to mutate to just stop worrying and let Nature take the wheel.

### "Individualistic" strategies (Prisoner's Dilemma strategies):
Strategies act on a history of past interactions in eight parameters: six "recent" parameters report the behavoir of self and opponent over the last three interactions, two "remote" parameters encode an average over all interactions, allowing a judgement of long-term trustworthiness.
Agents have a default predisposition for cooperation (self.cooperation_bias), which dominates in the first interaction. The eight parameters mentioned are given linear weights (self.linear_weights), and the two "remote" and six "recent" parameters of interaction history are weighed by means of a 2x2 and 6x6 symmetric matrices (self.remote_weights, self.recent_weights) using what I would like to call an "anti-quadratic form", i.e. instead of the standard quadratic form v^T W v, we calculate abs(v)^T W v, where past interactions are encoded in v with sign information (negative for defection, positive for cooperation). The "anti-quadratic" part allows for preservation of sign information. This is chosen over the more standard approach of encoding cooperation as 0 and defection as 1, or vice versa, because of the asymmetry involved in whichever action is encoded as zero giving no contribution to the overall outcome. Idk if this approach is at all known in game theory or if it has a name, but it is what has worked well for this particular task.
The outcome of this calculation, let's call it "cooperation propensity", is then fed into a linear slope function mapping values <-5 to 0 and >+5 to 1. This is intended to approximate a sigmoid (logistic) function, here called "pseudo-sigmoid". Values <-5 and >+5 result in deterministic behavior (defection or cooperation) while values in [-5,5] result in probabilistic behavior. It is thus possible to encode both deterministic and probabilistic strategies in agent genome.
The calculated value may be modified by "tribal" and "sexual" mechanics as described below before being fed into the pseudo-sigmoid function.

### "Tribal" modifier
Tribal or ethnic identity is encoded in an n-dimensional (IDENTITY_DIMENSIONS, default value 5) vector (self.identity). These values have no immediate expression in behavior, they just serve as markers of ethnic relatedness by means of calculating the Euclidean distance between two agents' identity vectors. The high dimensionality is just there to ensure that genetic drift doesn't create "relatedness" by accident. In addition, there is a gene encoding ingroup preference or xenophobia (self.ingroup_preference). Identity distance to interacting agent is weighed by ingroup preference in order to apply a bonus or penalty on the calculated cooperation propensity. 

### "Sexual" strategies
Sexual reproduction just averages male and female genomes (applying Gaussian noise mutation) and determines offspring sex at random. 
If sexual reproduction is enabled, there is a range of parameters modifying behavior in mating and in Prisoner's Dilemma interactions. First, there is a constant weight (self.male_bias) which is carried only by males and only inherited patrilineally, which shifts cooperation propensity for males, introducing basic sexual dimorphism. In addition, there is a parameter (self.chivalry) that shifts cooperation propensity specifically for males interacting with females (this parameter is in the genome of both males and females but only expresses behaviorally in males).
Additional parameters influence mating behavior to simulate sexual selection. Each female above a certain energy threshold (fertile females) is paired with every male in her Moore neigborhood for evaluation. Male mating behavior is modified by two parameters modelling investment in offspring (male_investment) and effort spent in courtship (mating_display). The male investment parameter modifies the reproduction cost for males in favor of females. This parameter is however invisible to female mating strategy. By contrast, the mating display parameter specifies an amount of additional energy expended by the male resulting in a higher value of male health as perceived by the female. Finally, ladys_choice is a 4-dimensional vector of weights expressing female mate preference. The four weights represent preference of "health", "fitness", "caring" and "kinship".

**"health"**: basically the current energy of the male suitor, but modified by male display behavior, weighed by the DISPLAY_EFFECTIVENESS constant.

**"fitness"**: measure of apparent evolutionary fitness of male, based on total score accumulated in Prisoner's Dilemma interaction over its lifespan somewhat modified by age and modified by number of offspring already produced.

**"caring"** is calculated from the Prisoners' Dilemma interaction history with this male, this is meant to be indirectly affected by the chivalry parameter, male agents with high chivalry will be perceived as more caring.

**"kinship"** is simply the distance measure between tribal identities.

All of these parameters, except for male_bias, are carried by both male and female agents, but male_investment, mating_display and chivalry is only expressed behaviorally in males while ladys_choice is only expressed behaviorally in females.
The scores of the male suitor in each of the four categories is weighed by the female choice parameters to calculate an overall score. This score is now modified by the male's energy value, weighed by BULLYING_EFFECTIVENESS, to simulate strong males brute-forcing their way to reproductive success by bullying away competitors, thus partially overriding female choice. The final result is fed into a softmax function to find the suitor destined for fatherhood.

