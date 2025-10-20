# Research Payoffs - Model of uncertainty and AI usage in research

# I modeled uncertainty and AI usage in research.
# We often face the intuition that useless things help research, and AI that easily completes tasks takes away that opportunity.

# This extends ps3's Problem 1 to real-world problems.
# Technically, I characterized function f to fit reality.

# More precisely, I characterized the uncertainty we face in research as follows.
# Technically, research involves both Conceptual and Mechanical work, and researchers don't know which type will occur until they do it.
# The intuition is like getting insights about equations while writing code, or finding connections to completely different fields while researching papers.

# each round i;
# Choose 0 (use AI) or 1 (do research yourself).

# AI is good at Mechanical work but not at Conceptual work.
# Conversely, researchers are good at Conceptual work but not at Mechanical work.

# Researchers use AI not only to automatically handle Mechanical tasks, but also to pursue perfection.

# To include this in the model, I made the following improvements.
# First, instead of directly getting payoff from the input choice, I made it so that two indicators are obtained: Knowledge gain and draft progress.
# AI is good at draft progress but not at Knowledge gain.
# Conversely, researchers are good at Knowledge gain but not at draft progress.
# Then, I express the payoff at that time as a linear combination of Knowledge gain and draft progress.
# Specifically, using lambda_t that takes values from 0 to lambda_1, I express it as follows.
# payoff = lambda_t * Knowledge gain + (1 - lambda_t) * draft progress
# Here, lambda_t is a parameter that takes values from 0 to 1.
# lambda_t changes smoothly over time, taking 0.3 in the first round and 0.8 in the last round, using exponential functions.
# The intuition is that the first half emphasizes understanding and the second half emphasizes completion.

# With full information, researchers can calculate the payoff for each different action every time.

# Besides research uncertainty, there are other reasons researchers use AI, such as anxiety about research progress.

# The final graph output is the total payoff, total understanding, and total completion. Each total is calculated per round.

# For Mechanical tasks: 
# Choose 0 (AI): Knowledge gain = 0.2, draft progress = 0.9
# Choose 1 (Human): Knowledge gain = 0.5, draft progress = 0.6

# For Conceptual tasks: 
# Choose 0 (AI): Knowledge gain = 0.1, draft progress = 0.4
# Choose 1 (Human): Knowledge gain = 0.8, draft progress = 0.3

# Apply inequality constraints here.
# Show that they hold within this.
# For conceptual tasks: human_knowledge > AI_knowledge, human_progress > AI_progress,
# For mechanical tasks: human_knowledge < AI_knowledge, human_progress < AI_progress,
# Strictly: conceptual_knowledge > mechanical_knowledge (human)
# Strictly: conceptual_progress < mechanical_progress (human)
# Within this, we can move freely

import numpy as np

class ResearchPayoffs:
    """
    Model of uncertainty and AI usage in research (knowledge accumulation version)
    
    Each round, researchers make the following choices:
    - 0: Use AI
    - 1: Do research yourself
    
    There are two types of research:
    - Mechanical: Mechanical work (AI is good at this)
    - Conceptual: Conceptual work (researchers are good at this)
    
    Two indicators are obtained from each choice:
    - Knowledge gain: Improvement in understanding (base value for that round)
    - Draft progress: Improvement in completion (depends on total knowledge)
    
    Knowledge accumulation effect:
    - Accumulated knowledge contributes to future progress improvement
    - knowledge_bonus = tanh(cumulative_knowledge * 0.1)
    - progress = base_progress * (1 + knowledge_bonus)
    
    Final payoff is a linear combination with weights that change over time:
    payoff = lambda_t * Knowledge_gain + (1 - lambda_t) * Draft_progress
    """
    
    def __init__(self, k=2, n=1000):
        self.k = k  # 2 choices: 0 (use AI) and 1 (do research yourself)
        self.n = n  # Total number of rounds
        self.cumulative_knowledge = 0
        self.cumulative_progress = 0
        self.cumulative_payoff = 0
        
        # Research type state transition (with stickiness)
        self.current_research_type = None  # Current research type
        self.stay_prob = 0.8  # Probability of maintaining the same state
        self.switch_prob = 0.2  # Probability of switching to another state
        
        # Mechanical tasks (work-oriented)
        # Generate values ensuring proper inequalities
        ai_k_m = np.random.uniform(0.1, 0.3)
        ai_p_m = np.random.uniform(0.8, 1.0)
        hu_k_m = np.random.uniform(ai_k_m + 0.1, 0.6)  # Ensure hu_k_m > ai_k_m
        hu_p_m = np.random.uniform(0.5, ai_p_m - 0.1)  # Ensure ai_p_m > hu_p_m

        # Conceptual tasks (concept-oriented)
        ai_k_c = np.random.uniform(0.05, ai_k_m - 0.05)  # Ensure ai_k_c < ai_k_m
        ai_p_c = np.random.uniform(0.2, hu_p_m - 0.1)    # Ensure hu_p_c > ai_p_c
        hu_k_c = np.random.uniform(hu_k_m + 0.1, 1.0)    # Ensure hu_k_c > hu_k_m
        hu_p_c = np.random.uniform(0.1, ai_p_c - 0.1)    # Ensure ai_p_c > hu_p_c
        
        # Safety check: ensure all inequalities are satisfied
        if not (hu_k_c > ai_k_c and hu_p_c > ai_p_c and 
                ai_k_m < hu_k_m and ai_p_m > hu_p_m and
                hu_k_c > hu_k_m and hu_p_c < hu_p_m):
            # If inequalities are not satisfied, regenerate with safer bounds
            print(f"Warning: Inequalities not satisfied, regenerating values...")
            print(f"  ai_k_m={ai_k_m:.3f}, hu_k_m={hu_k_m:.3f}, ai_k_c={ai_k_c:.3f}, hu_k_c={hu_k_c:.3f}")
            print(f"  ai_p_m={ai_p_m:.3f}, hu_p_m={hu_p_m:.3f}, ai_p_c={ai_p_c:.3f}, hu_p_c={hu_p_c:.3f}")
            
            ai_k_c = np.random.uniform(0.05, 0.2)
            ai_p_c = np.random.uniform(0.2, 0.4)
            hu_k_c = np.random.uniform(0.7, 1.0)
            hu_p_c = np.random.uniform(0.1, 0.3)

        # Effects of each choice (Knowledge gain, Draft progress)
        self.effects = {
            'mechanical': {
                'ai': (ai_k_m, ai_p_m),
                'human': (hu_k_m, hu_p_m)
            },
            'conceptual': {
                'ai': (ai_k_c, ai_p_c),
                'human': (hu_k_c, hu_p_c)
            }
        }
        
        # Weight parameter that changes over time
        # Initially emphasizes understanding (0.3), finally emphasizes completion (0.8)
        self.lambda_0 = 0.3
        self.lambda_1 = 0.8
        
    def _get_lambda(self, round_num):
        """Calculate weight parameter that changes over time"""
        # Changes smoothly using exponential function
        t = round_num / (self.n - 1) if self.n > 1 else 0
        return self.lambda_0 + (self.lambda_1 - self.lambda_0) * (1 - np.exp(-3 * t))
    
    def _determine_research_type(self):
        """Determine research type (state transition with stickiness)"""
        if self.current_research_type is None:
            # First round: decide randomly
            self.current_research_type = 'mechanical' if np.random.random() < 0.5 else 'conceptual'
        else:
            # Transition from previous state
            if np.random.random() < self.stay_prob:
                # 0.8 probability of maintaining the same state
                pass  # Use self.current_research_type as is
            else:
                # 0.2 probability of switching to another state
                self.current_research_type = 'conceptual' if self.current_research_type == 'mechanical' else 'mechanical'
        
        return self.current_research_type
    
    def _get_action_choice(self, action):
        """Convert selected action to string"""
        if action == 0:
            return 'ai'      # 0 = Use AI
        else:
            return 'human'   # 1 = Do research yourself
    
    def generate_payoffs(self, round_num):
        """
        Generate payoff for specified round
        Model where Knowledge gain accumulates and Draft progress depends on total knowledge
        
        Args:
            round_num: Current round number
            
        Returns:
            np.array: Payoff for each choice [AI usage payoff, self-research payoff]
        """
        # Determine research type
        research_type = self._determine_research_type()
        
        # Get basic effects for each choice
        ai_effects = self.effects[research_type]['ai']
        human_effects = self.effects[research_type]['human']
        
        # Calculate current weight parameter
        lambda_t = self._get_lambda(round_num)
        
        # Knowledge gain (base value - understanding improvement for that round)
        ai_knowledge = ai_effects[0]
        human_knowledge = human_effects[0]
        
        # Draft progress (depends on total knowledge)
        # Progress improvement through knowledge accumulation (non-linear, with upper limit)
        knowledge_bonus = np.tanh(self.cumulative_knowledge * 0.1)  # Range 0-1
        
        ai_progress = ai_effects[1] * (1 + knowledge_bonus)
        human_progress = human_effects[1] * (1 + knowledge_bonus)
        
        # Payoff calculation
        ai_payoff = lambda_t * ai_knowledge + (1 - lambda_t) * ai_progress
        human_payoff = lambda_t * human_knowledge + (1 - lambda_t) * human_progress
        
        return np.array([ai_payoff, human_payoff])
    
    def update_cumulative_stats(self, action, round_num):
        """
        Update cumulative statistics based on selected action
        Progress improvement model through knowledge accumulation
        
        Args:
            action: Selected action (0=use AI, 1=do research yourself)
            round_num: Current round number
        """
        # Re-get research type (already determined in generate_payoffs)
        research_type = self.current_research_type
        
        # Ensure research type is determined
        if research_type is None:
            research_type = self._determine_research_type()
        
        # Get basic effects of selected action
        if action == 0:  # Use AI
            base_knowledge, base_progress = self.effects[research_type]['ai']
        else:  # Do research yourself
            base_knowledge, base_progress = self.effects[research_type]['human']
        
        # Knowledge gain (understanding improvement for that round)
        knowledge_gain = base_knowledge
        
        # Draft progress (depends on total knowledge)
        # Progress improvement based on current total knowledge
        knowledge_bonus = np.tanh(self.cumulative_knowledge * 0.1)
        progress_gain = base_progress * (1 + knowledge_bonus)
        
        # Update cumulative values
        self.cumulative_knowledge += knowledge_gain
        self.cumulative_progress += progress_gain
        
        # Also calculate and accumulate payoff
        lambda_t = self._get_lambda(round_num)
        payoff = lambda_t * knowledge_gain + (1 - lambda_t) * progress_gain
        self.cumulative_payoff += payoff
    
    def get_cumulative_stats(self):
        """Get cumulative statistics"""
        return {
            'cumulative_knowledge': self.cumulative_knowledge,
            'cumulative_progress': self.cumulative_progress,
            'cumulative_payoff': self.cumulative_payoff
        }
    
    def reset(self):
        """Reset state"""
        self.cumulative_knowledge = 0
        self.cumulative_progress = 0
        self.cumulative_payoff = 0
        self.current_research_type = None  # Also reset research type