import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

from FCMNet import FCMNet
from alg_parameters import *


class Policy(object):
    """Build action policy of model."""

    def __init__(self, env, batch_size, sess=None):
        # Build  net
        self.obs = tf.placeholder(shape=(batch_size,) + (env.num_agents,) + (env.obs_shape,), dtype=tf.float32,
                                  name='ob')
        self.state = tf.placeholder(shape=(batch_size,) + (env.num_agents,) + (env.cent_state_shape,), dtype=tf.float32,
                                    name='state')
        self.critic_input_state_c = tf.placeholder(shape=(batch_size * env.num_agents,) + (CRITIC_LAYER2,),
                                                   dtype=tf.float32,
                                                   name='critic_state_c')
        self.critic_input_state_h = tf.placeholder(shape=(batch_size * env.num_agents,) + (CRITIC_LAYER2,),
                                                   dtype=tf.float32,
                                                   name='critic_state_h')
        self.actor_input_state_c = tf.placeholder(shape=(batch_size * env.num_agents,) + (ACTOR_LAYER2,),
                                                  dtype=tf.float32,
                                                  name='actor_state_c')
        self.actor_input_state_h = tf.placeholder(shape=(batch_size * env.num_agents,) + (ACTOR_LAYER2,),
                                                  dtype=tf.float32,
                                                  name='actor_state_h')
        critic_input_state = LSTMStateTuple(self.critic_input_state_c, self.critic_input_state_h)
        actor_input_state = LSTMStateTuple(self.actor_input_state_c, self.actor_input_state_h)
        network = FCMNet()
        v, self.critic_output_state = network.build_critic_network(self.state, critic_input_state)
        self.v = tf.squeeze(v)
        self.logits, self.actor_output_state = network.build_actor_network(self.obs,
                                                                           actor_input_state)  
        self.dist = tf.distributions.Categorical(logits=self.logits)
        self.ps = tf.nn.softmax(self.logits)
        self.sess = sess

    def step(self, observation, state, valid_action, critic_input_state_c, critic_input_state_h,
             actor_input_state_c, actor_input_state_h):
        actions = np.zeros((N_ENVS, N_AGENTS))
        feed_dict = {self.obs: observation, self.state: state, self.critic_input_state_c: critic_input_state_c,
                     self.critic_input_state_h: critic_input_state_h, self.actor_input_state_c: actor_input_state_c,
                     self.actor_input_state_h: actor_input_state_h}
        v, ps, critic_output_state, actor_output_state = self.sess.run([self.v, self.ps,
                                                                        self.critic_output_state,
                                                                        self.actor_output_state], feed_dict)
        ps[valid_action == 0.0] = 0.0
        ps /= np.expand_dims(np.sum(ps, axis=-1), -1)
        for i in range(N_ENVS):
            for j in range(N_AGENTS):
                actions[i, j] = np.random.choice(range(N_ACTIONS), p=ps[i, j])
        return actions, v, ps, critic_output_state, actor_output_state

    def value(self, state, critic_input_state_c, critic_input_state_h):
        feed_dict = {self.state: state, self.critic_input_state_c: critic_input_state_c,
                     self.critic_input_state_h: critic_input_state_h}
        return self.sess.run(self.v, feed_dict)

    def evalue(self, observation, valid_action, actor_input_state_c, actor_input_state_h):
        """Greedy action"""
        valid_action = np.array(valid_action, dtype=np.float)
        valid_action = np.expand_dims(valid_action, axis=0)
        feed_dict = {self.obs: observation, self.actor_input_state_c: actor_input_state_c,
                     self.actor_input_state_h: actor_input_state_h}
        ps, actor_output_state = self.sess.run([self.ps, self.actor_output_state], feed_dict)
        ps[valid_action == 0.0] = 0.0
        evalue_action = np.argmax(ps, axis=-1)

        return evalue_action, actor_output_state
