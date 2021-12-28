import numpy as np
import tensorflow as tf
from smac.env import StarCraft2Env

from alg_parameters import *
from env_parameters import env_args
from util import set_global_seeds


def model_act(obs, valid_a, actor_hidden_s_c, actor_hidden_s_h):
    valid_a = np.array(valid_a, dtype=np.float)
    obs = np.array(obs, dtype=np.float)
    obs = np.expand_dims(obs, axis=0)
    feed_dict = {obs_tensor: obs, actor_input_state_c_tensor: actor_hidden_s_c,
                 actor_input_state_h_tensor: actor_hidden_s_h}
    p, actor_hidden_s = sess.run([p_tensor, actor_output_state_tensor], feed_dict)
    p[0][valid_a == 0.0] = 0.0
    a = np.argmax(p, axis=-1)
    return a[0], actor_hidden_s


with tf.Session() as sess:
    # Restore graph
    saver = tf.train.import_meta_graph(r"./my_model/5m6m_FCMNet/final/final.meta")
    saver.restore(sess, tf.train.latest_checkpoint(r'./my_model/5m6m_FCMNet/final'))
    graph = tf.get_default_graph()
    obs_tensor = graph.get_tensor_by_name('ppo_model/ob_2:0')
    p_tensor = graph.get_tensor_by_name('ppo_model/Softmax_2:0')
    actor_input_state_c_tensor = graph.get_tensor_by_name('ppo_model/actor_state_c_2:0')
    actor_input_state_h_tensor = graph.get_tensor_by_name('ppo_model/actor_state_h_2:0')
    actor_output_state_tensor = (
        graph.get_tensor_by_name('ppo_model/actor_network_2/full_communication_layer/memory/memory/lstm_memory_cell/Add_1:0'),
        graph.get_tensor_by_name('ppo_model/actor_network_2/full_communication_layer/memory/memory/lstm_memory_cell/Mul_2:0'))

    # Setting up an evaluation environment
    env = StarCraft2Env(**env_args)
    env.seed(env_args['seed'] + 100 * N_ENVS)
    set_global_seeds(1234)

    # Running
    all_reward = []
    all_win_rate = []
    for i in range(10):
        eval_reward = []
        eval_info = []
        for j in range(EVALUE_EPISODES):
            env.reset()
            done = False
            episode_rewards = 0
            actor_hidden_state_c = np.zeros((N_AGENTS, ACTOR_LAYER2))
            actor_hidden_state_h = np.zeros((N_AGENTS, ACTOR_LAYER2))
            while not done:
                observation = env.get_obs()
                valid_action = env.get_avail_actions()
                action, actor_hidden_state = model_act(observation, valid_action, actor_hidden_state_c,
                                                       actor_hidden_state_h)
                actor_hidden_state_c, actor_hidden_state_h = actor_hidden_state
                r, done, info = env.step(action)
                episode_rewards += r
                # env.save_replay()
            eval_reward.append(episode_rewards)
            eval_info.append(info)
        all_win_rate.append(np.nanmean([item['battle_won'] for item in eval_info]))
        all_reward.append(np.nanmean(eval_reward))
    env.close()

    # recording
    win_rate_mean = round(np.nanmean(all_win_rate) * 100, 1)
    win_rate_std = round(np.std(all_win_rate) * 100, 1)
    reward_mean = round(np.nanmean(all_reward), 1)
    reward_std = round(np.std(all_reward), 1)
    print('mean of win rate:{}, std of win rate:{}, mean of episode reward:{}, std of episode reward:{}'.format(
        win_rate_mean, win_rate_std, reward_mean, reward_std))
