import os
import setproctitle
import tensorflow as tf

from alg_parameters import *
from env_wrapper import build_multiprocessing_env
from learner import learn
from util import get_session, save_state, set_global_seeds


def train():
    setproctitle.setproctitle('5m6m-StarCraft-' + EXPERIMENT_NAME + "@" + USER_NAME)
    env = build_multiprocessing_env(N_ENVS)
    learn(env=env)
    return env


def main():
    # Setting up tf environment
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=0,
                            inter_op_parallelism_threads=0)
    config.gpu_options.allow_growth = True
    get_session(config=config)
    set_global_seeds(SEED)

    # Key function
    env = train()

    # Save final model
    save_path = "my_model/" + EXPERIMENT_NAME + "/" + "final"
    os.makedirs(save_path, exist_ok=True)
    save_path += "/" + "final"
    save_state(save_path)

    env.close()


if __name__ == '__main__':
    main()
