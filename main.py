import datetime
import os
import setproctitle

import gym
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

from arguments import *
from buffer import ReplayBuffer
from env_wrapper import build_multiprocessing_env
from model import FCMNet
import environments


def linear_anneal(current_step, start=0.1, stop=1.0, steps=1e6):
    if current_step <= steps:
        eps = stop + (start - stop) * (1 - current_step / steps)
    else:
        eps = start
    return eps


def process_action(a, number=NUM_AGENTS, act_dim=ACTOR_OUTPUT_LEN):
    # Produce one-hot action
    one_hot_a = np.zeros((N_ENVS, number, act_dim))
    for i in range(N_ENVS):
        for j in range(number):
            one_hot_a[i, j, a[i, j]] = 1
    return one_hot_a


def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]


class ActorCriticEntropy(object):
    """Building actor and critic of multi-agent discrete SAC algorithm."""
    def __init__(self, sess):
        self.sess = sess
        self.model = FCMNet(sess)

        self.actor_input, self.mu, self.pi, self.action_probs, self.log_action_probs, self.actor_input_state_c, \
            self.actor_input_state_h, self.actor_output_state = self.create_actor_network(
                "actor_network")
        actor_network_params = tf.trainable_variables('actor_network')

        self.critic_input1, self.action1, self.q_logits1, self.q_a1 = self.create_critic_network("critic_network1")
        critic_network_params1 = tf.trainable_variables('critic_network1')

        self.target_critic_input1, self.target_action1, self.target_q_logits1, \
            self.target_q_a1 = self.create_critic_network("target_critic_network1")
        critic_target_network_params1 = tf.trainable_variables('target_critic_network1')

        self.critic_input2, self.action2, self.q_logits2, self.q_a2 = self.create_critic_network("critic_network2")
        critic_network_params2 = tf.trainable_variables('critic_network2')

        self.target_critic_inputs2, self.target_action2, self.target_q_logits2, \
            self.target_q_a2 = self.create_critic_network("target_critic_network2")
        critic_target_network_params2 = tf.trainable_variables('target_critic_network2')

        # Soft update target net
        with tf.name_scope("critic_update_target_network_params"):
            self.update_critic_target_network_params1 = [
                critic_target_network_params1[i].assign(tf.multiply(critic_network_params1[i], TAU)
                                                        + tf.multiply(critic_target_network_params1[i], 1. - TAU))
                for i in range(len(critic_target_network_params1))]
            self.update_critic_target_network_params2 = [
                critic_target_network_params2[i].assign(tf.multiply(critic_network_params2[i], TAU)
                                                        + tf.multiply(critic_target_network_params2[i], 1. - TAU))
                for i in range(len(critic_target_network_params2))]

        log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
        self.alpha = tf.exp(log_alpha)

        with tf.name_scope("critic_loss"):
            self.q_backup = tf.placeholder(tf.float32, (None, NUM_AGENTS), name="predicted_q_value")
            self.q1_loss = 0.5 * tf.reduce_mean((self.q_backup - self.q_a1) ** 2)
            self.q2_loss = 0.5 * tf.reduce_mean((self.q_backup - self.q_a2) ** 2)

        with tf.name_scope("actor_loss"):
            min_q_logits = tf.minimum(self.q_logits1, self.q_logits2)
            pi_backup = tf.reduce_sum(self.action_probs * (self.alpha * self.log_action_probs - min_q_logits), axis=-1)
            self.pi_loss = tf.reduce_mean(pi_backup)

        # Alpha loss for temperature parameter
        with tf.name_scope("alpha_loss"):
            max_target_entropy = tf.log(tf.cast(ACTOR_OUTPUT_LEN, tf.float32))
            self.target_entropy_prop = tf.placeholder(dtype=tf.float32, shape=())
            target_entropy = max_target_entropy * self.target_entropy_prop
            pi_entropy = -tf.reduce_sum(self.action_probs * self.log_action_probs, axis=-1)
            alpha_backup = tf.stop_gradient(target_entropy - pi_entropy)
            self.alpha_loss = -tf.reduce_mean(log_alpha * alpha_backup)

        # Training actor
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=ACTOR_LR)
        self.train_pi_op = pi_optimizer.minimize(self.pi_loss, var_list=actor_network_params)

        # Training critic
        value_optimizer = tf.train.AdamOptimizer(learning_rate=CRITIC_LR)
        self.train_value_op1 = value_optimizer.minimize(self.q1_loss, var_list=critic_network_params1)
        self.train_value_op2 = value_optimizer.minimize(self.q2_loss, var_list=critic_network_params2)

        # Training temperature parameter
        alpha_optimizer = tf.train.AdamOptimizer(learning_rate=ALPHA_LR)
        self.train_alpha_op = alpha_optimizer.minimize(self.alpha_loss, var_list=get_vars('log_alpha'))

    def create_actor_network(self, name):
        inputs = tf.placeholder(tf.float32, shape=(None, NUM_AGENTS, ACTOR_INPUT_LEN), name="actor_inputs")
        actor_input_state_c = tf.placeholder(shape=(None,) + (ACTOR_LAYER2,),
                                             dtype=tf.float32)
        actor_input_state_h = tf.placeholder(shape=(None,) + (ACTOR_LAYER2,),
                                             dtype=tf.float32)
        input_state = LSTMStateTuple(actor_input_state_c, actor_input_state_h)
        mu, pi, action_probs, log_action_probs, actor_output_state = self.model.actor_build_network(
            name, inputs, input_state)
        return inputs, mu, pi, action_probs, log_action_probs, actor_input_state_c, actor_input_state_h, \
            actor_output_state

    @staticmethod
    def calculate_z(z_input):
        p = np.random.rand(z_input.shape[0], z_input.shape[1])
        mask = np.asarray(np.less_equal(p, (1.0 + z_input) / 2.0), dtype=np.int)
        output = np.add(np.multiply(mask, 1.0 - z_input),
                        np.multiply(np.subtract(1.0, mask), -z_input - 1.0))
        return output

    def get_z(self, inputs):
        """Produce discrete data in the set {-1,1} based on the continuous output of the encoder."""
        output_1 = self.sess.run(
            [self.model.one_output_1, self.model.two_output_1, self.model.three_output_1,
             self.model.four_output_1, self.model.five_output_1],
            feed_dict={self.actor_input: inputs})
        output_1_zs = []
        for i in range(NUM_AGENTS):
            output_1_z = self.calculate_z(output_1[i])
            output_1_zs.append(output_1_z)

        output_2 = self.sess.run(
            [self.model.one_output_2, self.model.two_output_2, self.model.three_output_2,
             self.model.four_output_2, self.model.five_output_2],
            feed_dict={self.actor_input: inputs,
                       self.model.one_add_1: output_1_zs[0], self.model.two_add_1: output_1_zs[1],
                       self.model.three_add_1: output_1_zs[2],
                       self.model.four_add_1: output_1_zs[3], self.model.five_add_1: output_1_zs[4]})
        output_2_zs = []
        for i in range(NUM_AGENTS):
            output_2_z = self.calculate_z(output_2[i])
            output_2_zs.append(output_2_z)

        output_3 = self.sess.run(
            [self.model.one_output_3, self.model.two_output_3, self.model.three_output_3,
             self.model.four_output_3, self.model.five_output_3],
            feed_dict={self.actor_input: inputs,
                       self.model.one_add_1: output_1_zs[0], self.model.two_add_1: output_1_zs[1],
                       self.model.three_add_1: output_1_zs[2],
                       self.model.four_add_1: output_1_zs[3], self.model.five_add_1: output_1_zs[4],
                       self.model.one_add_2: output_2_zs[0], self.model.two_add_2: output_2_zs[1],
                       self.model.three_add_2: output_2_zs[2],
                       self.model.four_add_2: output_2_zs[3], self.model.five_add_2: output_2_zs[4]
                       })
        output_3_zs = []
        for i in range(NUM_AGENTS):
            output_3_z = self.calculate_z(output_3[i])
            output_3_zs.append(output_3_z)

        output_4 = self.sess.run(
            [self.model.one_output_4, self.model.two_output_4, self.model.three_output_4,
             self.model.four_output_4, self.model.five_output_4],
            feed_dict={self.actor_input: inputs,
                       self.model.one_add_1: output_1_zs[0], self.model.two_add_1: output_1_zs[1],
                       self.model.three_add_1: output_1_zs[2],
                       self.model.four_add_1: output_1_zs[3], self.model.five_add_1: output_1_zs[4],
                       self.model.one_add_2: output_2_zs[0], self.model.two_add_2: output_2_zs[1],
                       self.model.three_add_2: output_2_zs[2],
                       self.model.four_add_2: output_2_zs[3], self.model.five_add_2: output_2_zs[4],
                       self.model.one_add_3: output_3_zs[0], self.model.two_add_3: output_3_zs[1],
                       self.model.three_add_3: output_3_zs[2],
                       self.model.four_add_3: output_3_zs[3], self.model.five_add_3: output_3_zs[4]
                       })
        output_4_zs = []
        for i in range(NUM_AGENTS):
            output_4_z = self.calculate_z(output_4[i])
            output_4_zs.append(output_4_z)

        return output_1_zs, output_2_zs, output_3_zs, output_4_zs

    def actor_train(self, obs, state, action, z_1, z_2, z_3, z_4, target_entropy_prop,
                    actor_input_state_c, actor_input_state_h):
        return self.sess.run([self.pi_loss, self.train_pi_op, self.alpha_loss, self.train_alpha_op], feed_dict={
            self.actor_input: obs,
            self.critic_input1: state,
            self.critic_input2: state,
            self.action1: action,
            self.action2: action,
            self.model.one_add_1: z_1[0],
            self.model.two_add_1: z_1[1],
            self.model.three_add_1: z_1[2],
            self.model.four_add_1: z_1[3], self.model.five_add_1: z_1[4],
            self.model.one_add_2: z_2[0], self.model.two_add_2: z_2[1],
            self.model.three_add_2: z_2[2],
            self.model.four_add_2: z_2[3], self.model.five_add_2: z_2[4],
            self.model.one_add_3: z_3[0], self.model.two_add_3: z_3[1],
            self.model.three_add_3: z_3[2],
            self.model.four_add_3: z_3[3], self.model.five_add_3: z_3[4],
            self.model.one_add_4: z_4[0], self.model.two_add_4: z_4[1],
            self.model.three_add_4: z_4[2],
            self.model.four_add_4: z_4[3], self.model.five_add_4: z_4[4],
            self.actor_input_state_c: actor_input_state_c,
            self.actor_input_state_h: actor_input_state_h,
            self.target_entropy_prop: target_entropy_prop})

    def actor_step_predict(self, inputs, z_1, z_2, z_3, z_4, actor_input_state_c, actor_input_state_h):
        return self.sess.run([self.pi, self.actor_output_state], feed_dict={
            self.actor_input: inputs, self.model.one_add_1: z_1[0],
            self.model.two_add_1: z_1[1],
            self.model.three_add_1: z_1[2],
            self.model.four_add_1: z_1[3], self.model.five_add_1: z_1[4],
            self.model.one_add_2: z_2[0], self.model.two_add_2: z_2[1],
            self.model.three_add_2: z_2[2],
            self.model.four_add_2: z_2[3], self.model.five_add_2: z_2[4],
            self.model.one_add_3: z_3[0], self.model.two_add_3: z_3[1],
            self.model.three_add_3: z_3[2],
            self.model.four_add_3: z_3[3], self.model.five_add_3: z_3[4],
            self.model.one_add_4: z_4[0], self.model.two_add_4: z_4[1],
            self.model.three_add_4: z_4[2],
            self.model.four_add_4: z_4[3], self.model.five_add_4: z_4[4],
            self.actor_input_state_c: actor_input_state_c,
            self.actor_input_state_h: actor_input_state_h,
        })

    def actor_train_predict(self, inputs, z_1, z_2, z_3, z_4, actor_input_state_c, actor_input_state_h):
        return self.sess.run([self.action_probs, self.log_action_probs], feed_dict={
            self.actor_input: inputs, self.model.one_add_1: z_1[0],
            self.model.two_add_1: z_1[1],
            self.model.three_add_1: z_1[2],
            self.model.four_add_1: z_1[3], self.model.five_add_1: z_1[4],
            self.model.one_add_2: z_2[0], self.model.two_add_2: z_2[1],
            self.model.three_add_2: z_2[2],
            self.model.four_add_2: z_2[3], self.model.five_add_2: z_2[4],
            self.model.one_add_3: z_3[0], self.model.two_add_3: z_3[1],
            self.model.three_add_3: z_3[2],
            self.model.four_add_3: z_3[3], self.model.five_add_3: z_3[4],
            self.model.one_add_4: z_4[0], self.model.two_add_4: z_4[1],
            self.model.three_add_4: z_4[2],
            self.model.four_add_4: z_4[3], self.model.five_add_4: z_4[4],
            self.actor_input_state_c: actor_input_state_c,
            self.actor_input_state_h: actor_input_state_h,
        })

    def evalue(self, inputs, z_1, z_2, z_3, z_4, actor_input_state_c, actor_input_state_h):
        """Greedy action"""
        return self.sess.run([self.mu, self.actor_output_state], feed_dict={
            self.actor_input: inputs, self.model.one_add_1: z_1[0],
            self.model.two_add_1: z_1[1],
            self.model.three_add_1: z_1[2],
            self.model.four_add_1: z_1[3], self.model.five_add_1: z_1[4],
            self.model.one_add_2: z_2[0], self.model.two_add_2: z_2[1],
            self.model.three_add_2: z_2[2],
            self.model.four_add_2: z_2[3], self.model.five_add_2: z_2[4],
            self.model.one_add_3: z_3[0], self.model.two_add_3: z_3[1],
            self.model.three_add_3: z_3[2],
            self.model.four_add_3: z_3[3], self.model.five_add_3: z_3[4],
            self.model.one_add_4: z_4[0], self.model.two_add_4: z_4[1],
            self.model.three_add_4: z_4[2],
            self.model.four_add_4: z_4[3], self.model.five_add_4: z_4[4],
            self.actor_input_state_c: actor_input_state_c,
            self.actor_input_state_h: actor_input_state_h,
        })

    def update_target_network(self):
        self.sess.run([self.update_critic_target_network_params1, self.update_critic_target_network_params2])

    def create_critic_network(self, name):
        inputs = tf.placeholder(tf.float32, shape=(None, NUM_AGENTS, CRITIC_INPUT_LEN), name="critic_inputs")
        action = tf.placeholder(tf.float32, shape=(None, NUM_AGENTS, ACTOR_OUTPUT_LEN), name="critic_action")
        q_logits, q_a = self.model.critic_build_network(name, inputs, action, )
        return inputs, action, q_logits, q_a

    def critic_train(self, inputs, action, predicted_q_value):
        return self.sess.run(
            [self.q1_loss, self.train_value_op1, self.q_a1, self.q2_loss, self.train_value_op2, self.q_a2], feed_dict={
                self.critic_input1: inputs,
                self.q_backup: predicted_q_value,
                self.action1: action,
                self.critic_input2: inputs,
                self.action2: action,
            })

    def critic_predict_target(self, inputs, action):
        return self.sess.run([self.target_q_logits1, self.target_q_a1, self.target_q_logits2, self.target_q_a2],
                             feed_dict={
                                 self.target_critic_input1: inputs,
                                 self.target_action1: action,
                                 self.target_critic_inputs2: inputs,
                                 self.target_action2: action
                             })

    def get_alpha(self):
        return self.sess.run(self.alpha)


def train(sess, env, actor_critic_entropy):
    """Update model and record performance."""
    # Recording alg and env setting
    folder_name = 'summaries/' + EXPERIMENT_NAME + datetime.datetime.now().strftime('%d-%m-%y%H%M')
    global_summary = tf.summary.FileWriter(folder_name, sess.graph)
    txt_path = folder_name + '/' + EXPERIMENT_NAME + '.txt'
    with open(txt_path, "w") as file:
        file.write(str(alg_args))

    evalue_env = gym.make('MultiagentCatch-v0')

    # Initialize
    sess.run(tf.global_variables_initializer())
    actor_critic_entropy.update_target_network()
    replay_buffer = ReplayBuffer(BUFFER_SIZE, SEED)
    env.reset()
    obs, state = env.get_obs_state()
    actor_hidden_state_c = np.zeros((N_ENVS * NUM_AGENTS, ACTOR_LAYER2))
    actor_hidden_state_h = np.zeros((N_ENVS * NUM_AGENTS, ACTOR_LAYER2))
    num_episode = 0
    num_step = 0
    target_entropy_prop = linear_anneal(current_step=num_episode, start=TARGET_ENTROPY_START, stop=TARGET_ENTROPY_STOP,
                                        steps=TARGET_ENTROPY_STET)
    last_test_t = -EVALUE_INTERVAL - 1
    last_save_t = -SAVE_INTERVAL - 1

    while True:
        one_z, two_z, three_z, four_z = actor_critic_entropy.get_z(obs)
        action, actor_output_state = actor_critic_entropy.actor_step_predict(obs, one_z, two_z, three_z, four_z,
                                                                             actor_hidden_state_c, actor_hidden_state_h)
        reward, dones, infos = env.step(action)
        num_step += N_ENVS
        if num_step > N_MAX_STEPS:
            break
        true_index = np.argwhere(dones)
        if len(true_index) != 0:
            num_episode += len(true_index)
            # Initialize memory
            true_index = np.squeeze(true_index)
            target_entropy_prop = linear_anneal(current_step=num_episode, start=TARGET_ENTROPY_START,
                                                stop=TARGET_ENTROPY_STOP,
                                                steps=TARGET_ENTROPY_STET)
            actor_hidden_state_c = np.reshape(actor_hidden_state_c, (-1, NUM_AGENTS, ACTOR_LAYER2))
            actor_hidden_state_h = np.reshape(actor_hidden_state_h, (-1, NUM_AGENTS, ACTOR_LAYER2))

            actor_hidden_state_c[true_index] = np.zeros(actor_hidden_state_c[true_index].shape)
            actor_hidden_state_h[true_index] = np.zeros(actor_hidden_state_h[true_index].shape)

            actor_hidden_state_c = np.reshape(actor_hidden_state_c, (-1, ACTOR_LAYER2))
            actor_hidden_state_h = np.reshape(actor_hidden_state_h, (-1, ACTOR_LAYER2))

        obs2, state2 = env.get_obs_state()
        action = process_action(action)

        for i in range(N_ENVS):
            actor_hidden_state_c = np.reshape(actor_hidden_state_c, (-1, NUM_AGENTS, ACTOR_LAYER2))
            actor_hidden_state_h = np.reshape(actor_hidden_state_h, (-1, NUM_AGENTS, ACTOR_LAYER2))
            replay_buffer.add(obs[i], action[i], reward[i], dones[i], obs2[i], state[i], state2[i],
                              actor_hidden_state_c[i], actor_hidden_state_h[i])
        obs = obs2
        state = state2
        actor_hidden_state_c, actor_hidden_state_h = actor_output_state

        if replay_buffer.size() > 5 * MINIBATCH_SIZE:
            # Training
            o_batch, a_batch, r_batch, d_batch, next_o_batch, s_batch, next_s_batch, \
                state_c_batch, state_h_batch = replay_buffer.sample_batch(
                    MINIBATCH_SIZE)
            state_c_batch = np.reshape(state_c_batch, (-1, ACTOR_LAYER2))
            state_h_batch = np.reshape(state_h_batch, (-1, ACTOR_LAYER2))

            # Calculate targets
            one_next_z_batch, two_next_z_batch, three_next_z_batch, four_next_z_batch = actor_critic_entropy.get_z(
                next_o_batch)
            action_probs_next, log_action_probs_next = actor_critic_entropy.actor_train_predict(next_o_batch,
                                                                                                one_next_z_batch,
                                                                                                two_next_z_batch,
                                                                                                three_next_z_batch,
                                                                                                four_next_z_batch,
                                                                                                state_c_batch,
                                                                                                state_h_batch)
            target_q_logits1, target_q_a1, target_q_logits2, target_q_a2 = actor_critic_entropy.critic_predict_target(
                next_s_batch, a_batch)
            min_q_logits_targ = np.minimum(target_q_logits1, target_q_logits2)
            alpha = actor_critic_entropy.get_alpha()

            yi = np.empty((MINIBATCH_SIZE, NUM_AGENTS))
            for k in range(MINIBATCH_SIZE):
                if d_batch[k]:
                    yi[k] = r_batch[k]
                else:
                    yi[k] = r_batch[k] + GAMMA * np.sum(
                        action_probs_next[k] * (min_q_logits_targ[k] - alpha * log_action_probs_next[k]), axis=-1)

            one_z_batch, two_z_batch, three_z_batch, four_z_batch = actor_critic_entropy.get_z(o_batch)
            pi_loss, _, alpha_loss, _ = actor_critic_entropy.actor_train(o_batch, s_batch, a_batch,
                                                                         one_z_batch, two_z_batch, three_z_batch,
                                                                         four_z_batch,
                                                                         target_entropy_prop, state_c_batch,
                                                                         state_h_batch)
            critic1_loss, _, q1, critic2_loss, _, q2 = actor_critic_entropy.critic_train(s_batch, a_batch, yi)

            min_q = np.minimum(q1, q2)
            batch_max_q = np.amax(min_q)

            actor_critic_entropy.update_target_network()

            replay_buffer.clear()
            # Recording
            summary_step = tf.Summary()
            summary_step.value.add(tag='Perf_Step/Max_Q', simple_value=batch_max_q)
            summary_step.value.add(tag='Perf_Step/Reward', simple_value=np.mean(r_batch))
            summary_step.value.add(tag='Loss_Step/Pi_loss', simple_value=pi_loss)
            summary_step.value.add(tag='Loss_Step/Critic1_loss', simple_value=critic1_loss)
            summary_step.value.add(tag='Loss_Step/Critic2_loss', simple_value=critic2_loss)
            summary_step.value.add(tag='Loss_Step/Alpha_loss', simple_value=alpha_loss)
            global_summary.add_summary(summary_step, num_step)
            global_summary.flush()

            print('| Reward: {:.4f} |  Step: {:d} | Qmax: {:.4f} '.format(np.mean(r_batch), num_step, batch_max_q))

        if (num_step - last_test_t) / EVALUE_INTERVAL >= 1.0:
            # Evaluate model
            last_test_t = num_step
            eval_reward, eval_ep_len = evaluate(evalue_env, actor_critic_entropy)
            summary_len_step = tf.Summary()
            summary_len_step.value.add(tag='Perf_Step/Episode_len', simple_value=eval_ep_len)
            summary_len_step.value.add(tag='Perf_Eval_Step/Reward', simple_value=eval_reward)
            print('| Len: {:.4f} |  Step: {:d} '.format(eval_ep_len, num_step))
            global_summary.add_summary(summary_len_step, num_step)
            global_summary.flush()

        if (num_step - last_save_t) / SAVE_INTERVAL >= 1.0:
            # Save model
            last_save_t = num_step
            save_path = "my_model/" + EXPERIMENT_NAME + "/" + '%.5i' % last_save_t
            os.makedirs(save_path, exist_ok=True)
            print('Saving to', save_path)
            save_path += "/" + '%.5i' % last_save_t
            save_state(save_path)


def evaluate(evalue_env, model):
    """Evaluate model."""
    eval_reward = []
    eval_ep_len = []
    for _ in range(EVALUE_EPISODES):
        evalue_env.reset()
        terminal = False
        episode_reward = 0
        ep_len = 0
        actor_hidden_state_c = np.zeros((NUM_AGENTS, ACTOR_LAYER2))
        actor_hidden_state_h = np.zeros((NUM_AGENTS, ACTOR_LAYER2))
        while not terminal:
            obs, state = evalue_env.get_obs_state()
            obs = np.expand_dims(obs, axis=0)
            one_z, two_z, three_z, four_z = model.get_z(obs)
            actions, actor_hidden_state = model.evalue(obs, one_z, two_z, three_z, four_z, actor_hidden_state_c,
                                                       actor_hidden_state_h)
            actor_hidden_state_c, actor_hidden_state_h = actor_hidden_state
            reward, terminal, info = evalue_env.step(np.squeeze(actions))
            ep_len += 1
            episode_reward += reward
            if terminal:
                eval_reward.append(episode_reward)
                eval_ep_len.append(ep_len)
    eval_reward = np.nanmean(eval_reward)
    eval_ep_len = np.nanmean(eval_ep_len)
    return eval_reward, eval_ep_len


def get_session(config=None):
    """Get default session or create one with a given config"""
    sess = tf.get_default_session()
    if sess is None:
        sess = tf.InteractiveSession(config=config)
    return sess


def save_state(file_name):
    """save trained model"""
    saver = tf.train.Saver()
    sess = get_session()
    saver.save(sess, file_name)


def main():
    # Setting up tf environment
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    setproctitle.setproctitle('FCMNet-pathfinding-' + EXPERIMENT_NAME + "@" + USER_NAME)

    with tf.Session(config=config) as sess:
        env = build_multiprocessing_env(N_ENVS)

        actor_critic_entropy = ActorCriticEntropy(sess)

        # Key function
        train(sess, env, actor_critic_entropy)

        # Save final model
        save_path = "my_model/" + EXPERIMENT_NAME + "/" + "final"
        os.makedirs(save_path, exist_ok=True)
        save_path += "/" + "final"
        save_state(save_path)

        env.close()


if __name__ == '__main__':
    main()
