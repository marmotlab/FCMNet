""" Create a Multi-agent PPO with FCMNet algorithm.

Algorithm Parameters:
actor_lr :actor network learning rate.
critic_lr :critic network learning rate.
GAMMA :discount factor for reward.
LAM  : discount factor for advantage.
CLIP_RANGE : clip range for ratio.
MAX_GRAD_NORM : clip gradient.
ENTROPY_COEF : entropy discount factor.
VALUE_COEF : value loss discount factor.
POLICY_COEF : policy loss discount factor.
N_STEPS : each environment act step.
N_MINIBATCHES : number of minibatch in one train batch.
N_EPOCHS : number of reuse experience.
N_ENVS : Number of environment copies being run in parallel.
N_MAX_STEPS : max number of total step .
N_UPDATES: number of update .
BATCH_SIZE : size of each running.
MINIBATCH_SIZE: size of each minibatch.
LOSS_PROBABILITY: probability of losing message
SEED : random seed.
SAVE_INTERVAL :save model interval.
EVALUE_INTERVAL :evaluate model interval.
EVALUE_EPISODES: number of evaluation episodes.
 """

# Algorithm parameters
actor_lr = lambda f: f * 4e-4
critic_lr = lambda f: f * 4e-4
GAMMA = 0.99
LAM = 0.95
CLIP_RANGE = 0.2
MAX_GRAD_NORM = 15
ENTROPY_COEF = 0.01
VALUE_COEF = 1
POLICY_COEF = 1
N_STEPS = 2 ** 11
N_MINIBATCHES = 8
N_EPOCHS = 8
N_ENVS = 16
N_MAX_STEPS = 1e8
N_UPDATES = int(N_MAX_STEPS // (N_STEPS * N_ENVS))
BATCH_SIZE = int(N_STEPS * N_ENVS)
MINIBATCH_SIZE = int(N_STEPS * N_ENVS // N_MINIBATCHES)
LOSS_PROBABILITY = 0.03

#  Other parameters
SEED = 1234
ENV_SEED = 1
SAVE_INTERVAL = 500
EVALUE_INTERVAL = 100000
EVALUE_EPISODES = 1
EXPERIMENT_NAME = 'Pathfinding_FCMNet_Loss_Msg_3'
USER_NAME = 'Yutong'

# Environment parameters
N_AGENTS = 5
N_ACTIONS = 5
EPISODE_LEN = 1024

# Network parameters
ACTOR_LAYER1 = 2 ** 7
ACTOR_LAYER2 = 2 ** 6
ACTOR_LAYER3 = 5
CRITIC_LAYER1 = 2 ** 8
CRITIC_LAYER2 = 2 ** 6
CRITIC_LAYER3 = 1
ACTOR_INPUT_LEN = 12
CRITIC_INPUT_LEN = 36

alg_args = {'actor_lr ': actor_lr, 'critic_lr ': critic_lr, 'GAMMA': GAMMA, 'LAM': LAM, 'CLIPRANGE': CLIP_RANGE,
            'MAX_GRAD_NORM': MAX_GRAD_NORM, 'ENTROPY_COEF': ENTROPY_COEF, 'VALUE_COEF': VALUE_COEF,
            'POLICY_COEF': POLICY_COEF, 'N_STEPS': N_STEPS, 'N_MINIBATCHES': N_MINIBATCHES,
            'N_EPOCHS': N_EPOCHS, 'N_ENVS': N_ENVS, 'N_MAX_STEPS': N_MAX_STEPS,
            'N_UPDATES': N_UPDATES, 'MINIBATCH_SIZE': MINIBATCH_SIZE, 'SEED': SEED,'LOSS_PROBABILITY':LOSS_PROBABILITY,
            'SAVE_INTERVAL': SAVE_INTERVAL, 'EVALUE_INTERVAL': EVALUE_INTERVAL, 'EVALUE_EPISODES': EVALUE_EPISODES,
            'EXPERIMENT_NAME': EXPERIMENT_NAME,
            'USER_NAME': USER_NAME}
