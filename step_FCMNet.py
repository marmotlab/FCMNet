import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn import static_rnn
import random

from alg_parameters import *


class StepFCMNet(object):
    """Build actor and critic graph."""

    def build_actor_network(self, observation, input_state):
        """Building a multi-agent actor net"""
        with tf.variable_scope('actor_network', reuse=tf.AUTO_REUSE):
            outputs = self.shared_dense_layer("actor_layer1", observation, ACTOR_LAYER1, activation='tanh')
            outputs = tf.unstack(outputs, N_AGENTS, 1)

            lstm_cell_one = BasicLSTMCell(ACTOR_LAYER2, forget_bias=1.0,
                                          name="lstm_cell_one")
            lstm_cell_two = BasicLSTMCell(ACTOR_LAYER2, forget_bias=1.0,
                                          name="lstm_cell_two")
            lstm_cell_three = BasicLSTMCell(ACTOR_LAYER2, forget_bias=1.0,
                                            name="lstm_cell_three")
            lstm_cell_four = BasicLSTMCell(ACTOR_LAYER2, forget_bias=1.0,
                                           name="lstm_cell_four")
            lstm_cell_five = BasicLSTMCell(ACTOR_LAYER2, forget_bias=1.0,
                                           name="lstm_cell_five")
            lstm_memory_cell = BasicLSTMCell(ACTOR_LAYER2, forget_bias=1.0,
                                             name="lstm_memory_cell")
            # Build communication net
            outputs, output_state = self.actor_communication_layer(lstm_cell_one, lstm_cell_two, lstm_cell_three,
                                                                   lstm_cell_four, lstm_cell_five, lstm_memory_cell,
                                                                   input_state, outputs)
            outputs = tf.stack(outputs, 1)
            logits = self.shared_dense_layer("actor_layer2", outputs, ACTOR_LAYER3)
        return logits, output_state

    def build_critic_network(self, state, input_state):
        """Building a multi-agent critic net"""
        with tf.variable_scope('critic_network', reuse=tf.AUTO_REUSE):
            outputs = self.shared_dense_layer("critic_layer1", state, CRITIC_LAYER1, activation='tanh')
            outputs = tf.unstack(outputs, N_AGENTS, 1)

            lstm_cell_one = BasicLSTMCell(CRITIC_LAYER2, forget_bias=1.0,
                                          name="lstm_cell_one")
            lstm_cell_two = BasicLSTMCell(CRITIC_LAYER2, forget_bias=1.0,
                                          name="lstm_cell_two")
            lstm_cell_three = BasicLSTMCell(CRITIC_LAYER2, forget_bias=1.0,
                                            name="lstm_cell_three")
            lstm_cell_four = BasicLSTMCell(CRITIC_LAYER2, forget_bias=1.0,
                                           name="lstm_cell_four")
            lstm_cell_five = BasicLSTMCell(CRITIC_LAYER2, forget_bias=1.0,
                                           name="lstm_cell_five")
            lstm_memory_cell = BasicLSTMCell(CRITIC_LAYER2, forget_bias=1.0,
                                             name="lstm_memory_cell")
            # Build communication net
            outputs, output_state = self.critic_communication_layer(lstm_cell_one, lstm_cell_two, lstm_cell_three,
                                                                    lstm_cell_four, lstm_cell_five,
                                                                    lstm_memory_cell, input_state, outputs)
            outputs = tf.stack(outputs, 1)
            v = self.shared_dense_layer("critic_layer2", outputs, 1)
        return v, output_state

    @staticmethod
    def shared_dense_layer(name, observation, output_len, activation=None):
        """The weights of dense layer are shared."""
        all_outputs = []
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for j in range(N_AGENTS):
                agent_obs = observation[:, j]
                if activation == 'relu':
                    outputs = tf.layers.dense(agent_obs, output_len, name="dense", activation=tf.nn.relu,
                                              kernel_initializer=tf.orthogonal_initializer)
                elif activation == 'tanh':
                    outputs = tf.layers.dense(agent_obs, output_len, name="dense", activation=tf.nn.tanh,
                                              kernel_initializer=tf.orthogonal_initializer)
                else:
                    outputs = tf.layers.dense(agent_obs, output_len, name="dense",
                                              kernel_initializer=tf.orthogonal_initializer)
                all_outputs.append(outputs)
            all_outputs = tf.stack(all_outputs, 1)
        return all_outputs

    @staticmethod
    def static_rnn(cell, inputs, dtype=None, scope=None):
        outputs = []
        with vs.variable_scope(scope or "rnn") as varscope:
            if varscope.caching_device is None:
                varscope.set_caching_device(lambda op: op.device)

            first_input = inputs[0]

            if first_input.get_shape().ndims != 1:
                input_shape = first_input.get_shape().with_rank_at_least(2)
                fixed_batch_size = input_shape[0]
                flat_inputs = nest.flatten(inputs)
                for flat_input in flat_inputs:
                    input_shape = flat_input.get_shape().with_rank_at_least(2)
                    batch_size, input_size = input_shape[0], input_shape[1:]
                    fixed_batch_size.merge_with(batch_size)
                    for i, size in enumerate(input_size):
                        if size.value is None:
                            raise ValueError(
                                "Input size (dimension %d of inputs) must be accessible via "
                                "shape inference, but saw value None." % i)

            if fixed_batch_size.value:
                batch_size = fixed_batch_size.value

            if getattr(cell, "get_initial_state", None) is not None:
                state_0 = cell.get_initial_state(
                    inputs=None, batch_size=batch_size, dtype=dtype)

            (output_0, state_1) = cell(inputs[0], state_0)
            if random.random() > (1.0-LOSS_PROBABILITY):
                state_c_1 = tf.zeros(state_1.c.shape)
                state_h_1 = tf.zeros(state_1.h.shape)
                state_1 = tf.contrib.rnn.LSTMStateTuple(state_c_1, state_h_1)
            outputs.append(output_0)

            varscope.reuse_variables()
            (output_1, state_2) = cell(inputs[1], state_1)
            # Loss message with a certain probability
            if random.random() > (1.0-LOSS_PROBABILITY):
                state_c_2 = tf.zeros(state_2.c.shape)
                state_h_2 = tf.zeros(state_2.h.shape)
                state_2 = tf.contrib.rnn.LSTMStateTuple(state_c_2, state_h_2)
            outputs.append(output_1)

            varscope.reuse_variables()
            (output_2, state_3) = cell(inputs[2], state_2)
            if random.random() > (1.0-LOSS_PROBABILITY):
                state_c_3 = tf.zeros(state_3.c.shape)
                state_h_3 = tf.zeros(state_3.h.shape)
                state_3 = tf.contrib.rnn.LSTMStateTuple(state_c_3, state_h_3)
            outputs.append(output_2)

            varscope.reuse_variables()
            (output_3, state_4) = cell(inputs[3], state_3)
            if random.random() > (1.0-LOSS_PROBABILITY):
                state_c_4 = tf.zeros(state_4.c.shape)
                state_h_4 = tf.zeros(state_4.h.shape)
                state_4 = tf.contrib.rnn.LSTMStateTuple(state_c_4, state_h_4)
            outputs.append(output_3)

            varscope.reuse_variables()
            (output_4, state_5) = cell(inputs[4], state_4)
            outputs.append(output_4)
            return outputs

    def actor_communication_layer(self, cell_one, cell_two, cell_three, cell_four, cell_five, memory_cell,
                                  input_state, five_inputs):
        """Building communication layer"""
        with vs.variable_scope("full_communication_layer"):
            one_inputs = [five_inputs[1], five_inputs[2], five_inputs[3], five_inputs[4], five_inputs[0]]
            two_inputs = [five_inputs[0], five_inputs[2], five_inputs[3], five_inputs[4], five_inputs[1]]
            three_inputs = [five_inputs[0], five_inputs[1], five_inputs[3], five_inputs[4], five_inputs[2]]
            four_inputs = [five_inputs[0], five_inputs[1], five_inputs[2], five_inputs[4], five_inputs[3]]

            with vs.variable_scope("memory") as memory_scope:
                memory_inputs = tf.stack(five_inputs, 1)
                memory_inputs = [tf.reshape(memory_inputs, (-1, memory_inputs.shape[-1]))]

                output_memory, output_state = static_rnn(
                    cell=memory_cell,
                    inputs=memory_inputs,
                    scope=memory_scope,
                    initial_state=input_state,
                    dtype=tf.float32)

                output_memory = tf.reshape(output_memory[0], (-1, N_AGENTS, output_memory[0].shape[-1]))
                output_memory = [output_memory[:, 0], output_memory[:, 1], output_memory[:, 2], output_memory[:, 3],
                                 output_memory[:, 4]]

            with vs.variable_scope("one") as one_scope:
                output_one = self.static_rnn(
                    cell=cell_one,
                    inputs=one_inputs,
                    dtype=tf.float32,
                    scope=one_scope)

            with vs.variable_scope("two") as two_scope:
                output_two = self.static_rnn(
                    cell=cell_two,
                    inputs=two_inputs,
                    dtype=tf.float32,
                    scope=two_scope)

            with vs.variable_scope("three") as three_scope:
                output_three = self.static_rnn(
                    cell=cell_three,
                    inputs=three_inputs,
                    dtype=tf.float32,
                    scope=three_scope)

            with vs.variable_scope("four") as four_scope:
                output_four = self.static_rnn(
                    cell=cell_four,
                    inputs=four_inputs,
                    dtype=tf.float32,
                    scope=four_scope)

            with vs.variable_scope("five") as five_scope:
                output_five = self.static_rnn(
                    cell=cell_five,
                    inputs=five_inputs,
                    dtype=tf.float32,
                    scope=five_scope)

        final_output_one = [output_one[4], output_one[0], output_one[1], output_one[2], output_one[3]]
        final_output_two = [output_two[0], output_two[4], output_two[1], output_two[2], output_two[3]]
        final_output_three = [output_three[0], output_three[1], output_three[4], output_three[2], output_three[3]]
        final_output_four = [output_four[0], output_four[1], output_four[2], output_four[4], output_four[3]]

        flat_outputs = tuple(array_ops.concat([one, two, three, four, five, memory], 1)
                             for one, two, three, four, five, memory in
                             zip(final_output_one, final_output_two, final_output_three, final_output_four,
                                 output_five, output_memory))

        outputs = nest.pack_sequence_as(
            structure=output_one, flat_sequence=flat_outputs)

        return outputs, output_state

    @staticmethod
    def critic_communication_layer(cell_one, cell_two, cell_three, cell_four, cell_five, memory_cell,
                                   input_state, five_inputs):
        """Building communication layer"""
        with vs.variable_scope("full_communication_layer"):
            one_inputs = [five_inputs[1], five_inputs[2], five_inputs[3], five_inputs[4], five_inputs[0]]
            two_inputs = [five_inputs[0], five_inputs[2], five_inputs[3], five_inputs[4], five_inputs[1]]
            three_inputs = [five_inputs[0], five_inputs[1], five_inputs[3], five_inputs[4], five_inputs[2]]
            four_inputs = [five_inputs[0], five_inputs[1], five_inputs[2], five_inputs[4], five_inputs[3]]

            with vs.variable_scope("memory") as memory_scope:
                memory_inputs = tf.stack(five_inputs, 1)
                memory_inputs = [tf.reshape(memory_inputs, (-1, memory_inputs.shape[-1]))]

                output_memory, output_state = static_rnn(
                    cell=memory_cell,
                    inputs=memory_inputs,
                    scope=memory_scope,
                    initial_state=input_state,
                    dtype=tf.float32)

                output_memory = tf.reshape(output_memory[0], (-1, N_AGENTS, output_memory[0].shape[-1]))
                output_memory = [output_memory[:, 0], output_memory[:, 1], output_memory[:, 2], output_memory[:, 3],
                                 output_memory[:, 4]]

            with vs.variable_scope("one") as one_scope:
                output_one, _ = static_rnn(
                    cell=cell_one,
                    inputs=one_inputs,
                    dtype=tf.float32,
                    scope=one_scope)

            with vs.variable_scope("two") as two_scope:
                output_two, _ = static_rnn(
                    cell=cell_two,
                    inputs=two_inputs,
                    dtype=tf.float32,
                    scope=two_scope)

            with vs.variable_scope("three") as three_scope:
                output_three, _ = static_rnn(
                    cell=cell_three,
                    inputs=three_inputs,
                    dtype=tf.float32,
                    scope=three_scope)

            with vs.variable_scope("four") as four_scope:
                output_four, _ = static_rnn(
                    cell=cell_four,
                    inputs=four_inputs,
                    dtype=tf.float32,
                    scope=four_scope)

            with vs.variable_scope("five") as five_scope:
                output_five, _ = static_rnn(
                    cell=cell_five,
                    inputs=five_inputs,
                    dtype=tf.float32,
                    scope=five_scope)

        final_output_one = [output_one[4], output_one[0], output_one[1], output_one[2], output_one[3]]
        final_output_two = [output_two[0], output_two[4], output_two[1], output_two[2], output_two[3]]
        final_output_three = [output_three[0], output_three[1], output_three[4], output_three[2], output_three[3]]
        final_output_four = [output_four[0], output_four[1], output_four[2], output_four[4], output_four[3]]

        flat_outputs = tuple(array_ops.concat([one, two, three, four, five, memory], 1)
                             for one, two, three, four, five, memory in
                             zip(final_output_one, final_output_two, final_output_three, final_output_four,
                                 output_five, output_memory))

        outputs = nest.pack_sequence_as(
            structure=output_one, flat_sequence=flat_outputs)

        return outputs, output_state
