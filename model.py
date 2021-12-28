import numpy as np
import tensorflow as tf

from alg_parameters import *
from policy import Policy
from util import get_session, initialize


class Model(object):
    """Build tensor graph and calculate flow."""

    def __init__(self, env):
        self.sess = get_session()

        with tf.variable_scope('ppo_model', reuse=tf.AUTO_REUSE):
            act_model = Policy(env, N_ENVS, self.sess)
            self.train_model = train_model = Policy(env, MINIBATCH_SIZE, self.sess)
            evalue_model = Policy(env, 1, self.sess)

        # Create placeholders
        self.action = tf.placeholder(tf.int32, [MINIBATCH_SIZE, env.num_agents])
        self.advantage = tf.placeholder(tf.float32, [MINIBATCH_SIZE, env.num_agents])
        self.returns = tf.placeholder(tf.float32, [MINIBATCH_SIZE, env.num_agents])
        # Keep track of old actor
        self.old_logp = tf.placeholder(tf.float32, [MINIBATCH_SIZE, env.num_agents])
        # Keep track of old critic
        self.old_v = tf.placeholder(tf.float32, [MINIBATCH_SIZE, env.num_agents])

        self.actor_lr = tf.placeholder(tf.float32, [])
        self.critic_lr = tf.placeholder(tf.float32, [])
        self.clip_range = tf.placeholder(tf.float32, [])

        new_logp = train_model.dist.log_prob(self.action)
        ratio = tf.exp(new_logp - self.old_logp)
        # Entropy
        entropy = tf.reduce_mean(train_model.dist.entropy())

        # Critic loss
        v_pred = train_model.v
        v_pred_clipped = self.old_v + tf.clip_by_value(train_model.v - self.old_v, - self.clip_range,
                                                       self.clip_range)
        value_losses1 = tf.square(v_pred - self.returns)
        value_losses2 = tf.square(v_pred_clipped - self.returns)
        critic_loss = .5 * tf.reduce_mean(tf.maximum(value_losses1, value_losses2))

        # Actor loss
        ratio = tf.squeeze(ratio)
        policy_losses = -self.advantage * ratio
        policy_losses2 = -self.advantage * tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        policy_loss = tf.reduce_mean(tf.maximum(policy_losses, policy_losses2))
        actor_loss = policy_loss - entropy * ENTROPY_COEF

        clip_frac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.clip_range)))

        # Training actor
        actor_params = tf.trainable_variables('ppo_model/actor_network')
        actor_trainer = tf.train.AdamOptimizer(learning_rate=self.actor_lr, epsilon=1e-5)
        actor_grads_and_var = actor_trainer.compute_gradients(actor_loss, actor_params)
        actor_grads, actor_var = zip(*actor_grads_and_var)
        actor_grads, actor_grad_norm = tf.clip_by_global_norm(actor_grads, MAX_GRAD_NORM)
        actor_grads_and_var = list(zip(actor_grads, actor_var))

        #  Training critic
        critic_params = tf.trainable_variables('ppo_model/critic_network')
        critic_trainer = tf.train.AdamOptimizer(learning_rate=self.critic_lr, epsilon=1e-5)
        critic_grads_and_var = critic_trainer.compute_gradients(critic_loss, critic_params)
        critic_grads, critic_var = zip(*critic_grads_and_var)
        critic_grads, critic_grad_norm = tf.clip_by_global_norm(critic_grads, MAX_GRAD_NORM)
        critic_grads_and_var = list(zip(critic_grads, critic_var))

        self.actor_train_op = actor_trainer.apply_gradients(actor_grads_and_var)
        self.critic_train_op = critic_trainer.apply_gradients(critic_grads_and_var)

        self.loss_names = ['actor_loss', 'policy_entropy', 'policy_loss', 'value_loss', 'clipfrac', 'actor_grad_norm',
                           'critic_grad_norm']
        self.stats_list = [actor_loss, entropy, policy_loss, critic_loss, clip_frac, actor_grad_norm, critic_grad_norm]

        self.step = act_model.step
        self.value = act_model.value
        self.evalue = evalue_model.evalue

        initialize()

    def train(self, a_lr, c_lr, cliprange, obs, state, returns, values, old_logp, action,
              critic_input_state_c, critic_input_state_h,
              actor_input_state_c, actor_input_state_h
              ):
        actor_input_state_c = np.reshape(actor_input_state_c, (-1, ACTOR_LAYER2))
        actor_input_state_h = np.reshape(actor_input_state_h, (-1, ACTOR_LAYER2))
        critic_input_state_c = np.reshape(critic_input_state_c, (-1, CRITIC_LAYER2))
        critic_input_state_h = np.reshape(critic_input_state_h, (-1, CRITIC_LAYER2))

        advantage = returns - values
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        td_map = {
            self.train_model.obs: obs,
            self.train_model.state: state,
            self.advantage: advantage,
            self.returns: returns,
            self.actor_lr: a_lr,
            self.critic_lr: c_lr,
            self.clip_range: cliprange,
            self.old_logp: old_logp,
            self.old_v: values,
            self.action: action,
            self.train_model.critic_input_state_c: critic_input_state_c,
            self.train_model.critic_input_state_h: critic_input_state_h,
            self.train_model.actor_input_state_c: actor_input_state_c,
            self.train_model.actor_input_state_h: actor_input_state_h,
        }

        state = self.sess.run(self.stats_list + [self.actor_train_op, self.critic_train_op], td_map)[:-2]
        return state
