import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

from train_FCMNet import TrainFCMNet
from alg_parameters import *


class TrainPolicy(object):
    """Build action policy of model."""

    def __init__(self, env, batch_size, sess=None):
        # Build  net
        self.obs = tf.placeholder(shape=(batch_size,) + (env.num_agents,) + (env.obs_shape,), dtype=tf.float32,
                                  name='ob')
        self.state = tf.placeholder(shape=(batch_size,) + (env.num_agents,) + (env.state_shape,), dtype=tf.float32,
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
        network = TrainFCMNet()
        v, self.critic_output_state = network.build_critic_network(self.state, critic_input_state)
        self.v = tf.squeeze(v)
        self.logits, self.actor_output_state = network.build_actor_network(self.obs,
                                                                           actor_input_state)
        self.dist = tf.distributions.Categorical(logits=self.logits)
        self.action = self.dist.sample()
        self.log_p = self.dist.log_prob(self.action)
        self.evalue_action = self.dist.mode()
        self.sess = sess

    def step(self, observation, state, critic_input_state_c, critic_input_state_h,
             actor_input_state_c, actor_input_state_h):
        feed_dict = {self.obs: observation, self.state: state, self.critic_input_state_c: critic_input_state_c,
                     self.critic_input_state_h: critic_input_state_h, self.actor_input_state_c: actor_input_state_c,
                     self.actor_input_state_h: actor_input_state_h}
        v, log_p, action, critic_output_state, actor_output_state = self.sess.run([self.v, self.log_p, self.action,
                                                                                   self.critic_output_state,
                                                                                   self.actor_output_state], feed_dict)
        return v, log_p, action, critic_output_state, actor_output_state

    def value(self, state, critic_input_state_c, critic_input_state_h):
        feed_dict = {self.state: state, self.critic_input_state_c: critic_input_state_c,
                     self.critic_input_state_h: critic_input_state_h}
        return self.sess.run(self.v, feed_dict)

    def evalue(self, observation, actor_input_state_c, actor_input_state_h):
        """Greedy action"""
        feed_dict = {self.obs: observation, self.actor_input_state_c: actor_input_state_c,
                     self.actor_input_state_h: actor_input_state_h}
        evalue_action, actor_output_state = self.sess.run([self.evalue_action, self.actor_output_state], feed_dict)
        return evalue_action, actor_output_state
