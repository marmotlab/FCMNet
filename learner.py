import datetime
import time
import os

import gym
import numpy as np
import tensorflow as tf
import random

from alg_parameters import *
from model import Model
from runner import Runner
from util import get_session, save_state
import environments


def learn(env):
    """Update model and record performance."""
    evalue_env = gym.make('MultiagentCatch-v0')

    sess = get_session()

    # Recording alg and env setting
    folder_name = 'summaries/' + EXPERIMENT_NAME + datetime.datetime.now().strftime('%d-%m-%y%H%M')
    global_summary = tf.summary.FileWriter(folder_name, sess.graph)
    txt_path = folder_name + '/' + EXPERIMENT_NAME + '.txt'
    with open(txt_path, "w") as file:
        file.write(str(alg_args))

    model = Model(env=env)
    runner = Runner(env=env, model=model)

    num_episodes = 0
    last_evaluation_t = -EVALUE_INTERVAL - 1
    start_time = time.perf_counter()

    for update in range(1, N_UPDATES + 1):
        frac = 1.0 - (update - 1.0) / N_UPDATES
        # Calculate learning rates
        actorlrnow = actor_lr(frac)
        criticlrnow = critic_lr(frac)

        # Get experience from multiple environments
        obs, state, returns, values, logp, actions, mb_critic_hidden_state_c, mb_critic_hidden_state_h, \
            mb_actor_hidden_state_c, mb_actor_hidden_state_h, ep_infos = runner.run()

        # Training
        mb_loss = []
        inds = np.arange(BATCH_SIZE)
        for _ in range(N_EPOCHS):
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = inds[start:end]
                slices = (arr[mb_inds] for arr in (obs, state, returns, values, logp, actions, mb_critic_hidden_state_c,
                                                   mb_critic_hidden_state_h, mb_actor_hidden_state_c,
                                                   mb_actor_hidden_state_h))
                mb_loss.append(model.train(actorlrnow, criticlrnow, CLIP_RANGE, *slices))

        # Recording
        loss_vals = np.nanmean(mb_loss, axis=0)
        num_episodes += len(ep_infos)
        summary_episodes = tf.Summary()
        summary_step = tf.Summary()
        performance_r = np.nanmean([items['episode_reward'] for items in ep_infos])
        performance_len = np.nanmean([items['episode_len'] for items in ep_infos])

        summary_episodes.value.add(tag='Perf/Reward', simple_value=performance_r)
        summary_episodes.value.add(tag='Perf/Episode_len', simple_value=performance_len)
        summary_step.value.add(tag='Perf_Step/Reward_Step', simple_value=performance_r)
        summary_step.value.add(tag='Perf_Step/Episode_len_Step', simple_value=performance_len)

        for (val, name) in zip(loss_vals, model.loss_names):
            if name == 'actor_grad_norm' or name == 'critic_grad_norm':
                summary_episodes.value.add(tag='grad/' + name, simple_value=val)
                summary_step.value.add(tag='grad_step/' + name, simple_value=val)
            else:
                summary_episodes.value.add(tag='loss/' + name, simple_value=val)
                summary_step.value.add(tag='loss_step/' + name, simple_value=val)

        global_summary.add_summary(summary_episodes, num_episodes)
        global_summary.add_summary(summary_step, update * BATCH_SIZE)
        global_summary.flush()

        if update % 10 == 0:
            print('update: {}, episode: {}, step: {}, episode reward: {}, episode len: {},'
                  .format(update, num_episodes, update * BATCH_SIZE, performance_r, performance_len))

        if (update * BATCH_SIZE - last_evaluation_t) / EVALUE_INTERVAL >= 1.0:
            # Evaluate model
            last_evaluation_t = update * BATCH_SIZE
            summary_eval_step = tf.Summary()
            summary_eval_episode = tf.Summary()
            eval_reward, eval_ep_len = evaluate(evalue_env, model)
            # Recording
            summary_eval_step.value.add(tag='Perf_evaluate_step/Reward', simple_value=eval_reward)
            summary_eval_step.value.add(tag='Perf_evaluate_step/Episode_len', simple_value=eval_ep_len)
            summary_eval_episode.value.add(tag='Perf_evaluate_episode/Reward', simple_value=eval_reward)
            summary_eval_episode.value.add(tag='Perf_evaluate_episode/Episode_len', simple_value=eval_ep_len)
            global_summary.add_summary(summary_eval_step, update * BATCH_SIZE)
            global_summary.add_summary(summary_eval_episode, num_episodes)
            global_summary.flush()

        if update % SAVE_INTERVAL == 0:
            t_now = time.perf_counter()
            print('consume time', t_now - start_time)
            save_path = "my_model/" + EXPERIMENT_NAME + "/" + '%.5i' % update
            os.makedirs(save_path, exist_ok=True)
            print('Saving to', save_path)
            save_path += "/" + '%.5i' % update
            save_state(save_path)

    evalue_env.close()


def evaluate(evalue_env, model):
    """Evaluate model."""
    eval_reward = []
    eval_ep_len = []
    for _ in range(EVALUE_EPISODES):
        evalue_env.reset()
        terminal = False
        episode_reward = 0
        ep_len = 0
        actor_hidden_state_c = np.zeros((N_AGENTS, ACTOR_LAYER2))
        actor_hidden_state_h = np.zeros((N_AGENTS, ACTOR_LAYER2))
        while not terminal:
            obs, state = evalue_env.get_obs_state()
            obs = np.expand_dims(obs, axis=0)
            # Due to parameter sharing, changing the input order equal to changing the connection order
            step_order = random.sample(range(0, 5), 5)
            changed_obs = [obs[:, step_order[0], :], obs[:, step_order[1], :], obs[:, step_order[2], :],
                           obs[:, step_order[3], :], obs[:, step_order[4], :]]
            changed_obs = np.stack(changed_obs, axis=1)
            changed_actions, actor_hidden_state = model.evalue(changed_obs, actor_hidden_state_c, actor_hidden_state_h)
            actor_hidden_state_c, actor_hidden_state_h = actor_hidden_state
            actions = np.zeros(changed_actions.shape)
            actions[:, step_order[0]], actions[:, step_order[1]], actions[:, step_order[2]], \
                actions[:, step_order[3]], actions[:, step_order[4]] = \
                changed_actions[:, 0], changed_actions[:, 1], changed_actions[:, 2], changed_actions[:, 3],\
                changed_actions[:, 4]
            reward, terminal, info = evalue_env.step(np.squeeze(actions))
            ep_len += 1
            episode_reward += reward
            if terminal:
                eval_reward.append(episode_reward)
                eval_ep_len.append(ep_len)
    eval_reward = np.nanmean(eval_reward)
    eval_ep_len = np.nanmean(eval_ep_len)
    return eval_reward, eval_ep_len
