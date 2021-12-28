import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.python.util import nest

from arguments import *


class FCMNet:
    """Build actor and critic graph. The messages transferred between actors are binarized."""
    def __init__(self, sess):
        self.sess = sess
        self.one_output_1, self.one_add_1, self.one_output_2, self.one_add_2, self.one_output_3, self.one_add_3, \
            self.one_output_4, self.one_add_4, self.two_output_1, self.two_add_1, self.two_output_2, self.two_add_2, \
            self.two_output_3, self.two_add_3, self.two_output_4, self.two_add_4, self.three_output_1, \
            self.three_add_1, self.three_output_2, self.three_add_2, self.three_output_3, self.three_add_3, \
            self.three_output_4, self.three_add_4, self.four_output_1, self.four_add_1, self.four_output_2, \
            self.four_add_2, self.four_output_3, self.four_add_3, self.four_output_4, self.four_add_4,\
            self.five_output_1, self.five_add_1, self.five_output_2, self.five_add_2, self.five_output_3, \
            self.five_add_3, self.five_output_4, self.five_add_4 = [None for _ in range(40)]

    @staticmethod
    def encoder(state_ch):
        """Encoding and decoding messages"""
        with tf.variable_scope("binary", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
                encode_layer = tf.layers.dense(state_ch, ENCODING_LAYER1, activation=tf.nn.relu,
                                               name="encode_layer_1")
                encoder_output = tf.layers.dense(encode_layer, ENCODING_OUTPUT, name="encode_layer_2",
                                                 activation=tf.nn.tanh)
                add_ph = tf.placeholder(shape=(None, ENCODING_OUTPUT), dtype=tf.float32)
                # Binarized
                message = encoder_output + add_ph
                decode_layer = tf.layers.dense(message, ENCODING_LAYER1, name="decode_layer_1",
                                               activation=tf.nn.relu)
                decode_output = tf.layers.dense(decode_layer, ENCODING_INPUT, name="decode_layer_2")
        return decode_output, encoder_output, add_ph

    def static_rnn(self, cell, inputs, dtype=None, scope=None):
        """Binarized communication channel"""
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
            batch_size = array_ops.shape(first_input)[0]

            state = cell.get_initial_state(inputs=None, batch_size=batch_size, dtype=dtype)

            (output_1, state_o_1) = cell(inputs[0], state)
            outputs.append(output_1)
            state_c_1 = state_o_1.c
            state_h_1 = state_o_1.h
            state_ch = tf.concat([state_c_1, state_h_1], axis=-1)
            decode_output, encoder_output_1, add_ph_1 = self.encoder(state_ch)
            state_c_1, state_h_1 = tf.split(decode_output, 2, axis=-1)
            state_1 = tf.contrib.rnn.LSTMStateTuple(state_c_1, state_h_1)

            varscope.reuse_variables()
            (output_2, state_o_2) = cell(inputs[1], state_1)
            outputs.append(output_2)
            state_c_2 = state_o_2.c
            state_h_2 = state_o_2.h
            state_ch = tf.concat([state_c_2, state_h_2], axis=-1)
            decode_output, encoder_output_2, add_ph_2 = self.encoder(state_ch)
            state_c_2, state_h_2 = tf.split(decode_output, 2, axis=-1)
            state_2 = tf.contrib.rnn.LSTMStateTuple(state_c_2, state_h_2)

            varscope.reuse_variables()
            (output_3, state_o_3) = cell(inputs[2], state_2)
            outputs.append(output_3)
            state_c_3 = state_o_3.c
            state_h_3 = state_o_3.h
            state_ch = tf.concat([state_c_3, state_h_3], axis=-1)
            decode_output, encoder_output_3, add_ph_3 = self.encoder(state_ch)
            state_c_3, state_h_3 = tf.split(decode_output, 2, axis=-1)
            state_3 = tf.contrib.rnn.LSTMStateTuple(state_c_3, state_h_3)

            varscope.reuse_variables()
            (output_4, state_o_4) = cell(inputs[3], state_3)
            outputs.append(output_4)
            state_c_4 = state_o_4.c
            state_h_4 = state_o_4.h
            state_ch = tf.concat([state_c_4, state_h_4], axis=-1)
            decode_output, encoder_output_4, add_ph_4 = self.encoder(state_ch)
            state_c_4, state_h_4 = tf.split(decode_output, 2, axis=-1)
            state_4 = tf.contrib.rnn.LSTMStateTuple(state_c_4, state_h_4)

            varscope.reuse_variables()
            (output_5, state_o_5) = cell(inputs[4], state_4)
            outputs.append(output_5)

            return outputs, encoder_output_1, add_ph_1, encoder_output_2, add_ph_2, encoder_output_3, add_ph_3, \
                encoder_output_4, add_ph_4

    def actor_comm(self, cell_one, cell_two, cell_three, cell_four, cell_five, memory_cell, input_state, five_inputs):
        """Building communication layer with binary messages"""
        with vs.variable_scope("full_communication_channels"):
            one_inputs = [five_inputs[1], five_inputs[2], five_inputs[3], five_inputs[4], five_inputs[0]]
            two_inputs = [five_inputs[0], five_inputs[2], five_inputs[3], five_inputs[4], five_inputs[1]]
            three_inputs = [five_inputs[0], five_inputs[1], five_inputs[3], five_inputs[4], five_inputs[2]]
            four_inputs = [five_inputs[0], five_inputs[1], five_inputs[2], five_inputs[4], five_inputs[3]]

            with vs.variable_scope("memory") as memory_scope:
                memory_inputs = tf.stack(five_inputs, 1)
                memory_inputs = [tf.reshape(memory_inputs, (-1, memory_inputs.shape[-1]))]
                output_memory, output_state = tf.contrib.rnn.static_rnn(
                    cell=memory_cell,
                    inputs=memory_inputs,
                    scope=memory_scope,
                    initial_state=input_state,
                    dtype=tf.float32)

                output_memory = tf.reshape(output_memory[0], (-1, NUM_AGENTS, output_memory[0].shape[-1]))
                output_memory = [output_memory[:, 0], output_memory[:, 1], output_memory[:, 2], output_memory[:, 3],
                                 output_memory[:, 4]]

            with vs.variable_scope("one") as one_scope:
                output_one, self.one_output_1, self.one_add_1, self.one_output_2, self.one_add_2, self.one_output_3, \
                    self.one_add_3, self.one_output_4, self.one_add_4 = self.static_rnn(
                        cell=cell_one,
                        inputs=one_inputs,
                        dtype=tf.float32,
                        scope=one_scope)

            with vs.variable_scope("two") as two_scope:
                output_two, self.two_output_1, self.two_add_1, self.two_output_2, self.two_add_2, self.two_output_3, \
                    self.two_add_3, self.two_output_4, self.two_add_4 = self.static_rnn(
                        cell=cell_two,
                        inputs=two_inputs,
                        dtype=tf.float32,
                        scope=two_scope)

            with vs.variable_scope("three") as three_scope:
                output_three, self.three_output_1, self.three_add_1, self.three_output_2, self.three_add_2, \
                    self.three_output_3, self.three_add_3, self.three_output_4, self.three_add_4 = self.static_rnn(
                        cell=cell_three,
                        inputs=three_inputs,
                        dtype=tf.float32,
                        scope=three_scope)

            with vs.variable_scope("four") as four_scope:
                output_four, self.four_output_1, self.four_add_1, self.four_output_2, self.four_add_2, \
                    self.four_output_3, self.four_add_3, self.four_output_4, self.four_add_4 = self.static_rnn(
                        cell=cell_four,
                        inputs=four_inputs,
                        dtype=tf.float32,
                        scope=four_scope)

            with vs.variable_scope("five") as five_scope:
                output_five, self.five_output_1, self.five_add_1, self.five_output_2, self.five_add_2, \
                    self.five_output_3, self.five_add_3, self.five_output_4, self.five_add_4 = self.static_rnn(
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
    def critic_comm(cell_one, cell_two, cell_three, cell_four, cell_five, five_inputs):
        """Building communication layer with real value messages"""
        with vs.variable_scope("full_communication_layer"):
            one_inputs = [five_inputs[1], five_inputs[2], five_inputs[3], five_inputs[4], five_inputs[0]]
            two_inputs = [five_inputs[0], five_inputs[2], five_inputs[3], five_inputs[4], five_inputs[1]]
            three_inputs = [five_inputs[0], five_inputs[1], five_inputs[3], five_inputs[4], five_inputs[2]]
            four_inputs = [five_inputs[0], five_inputs[1], five_inputs[2], five_inputs[4], five_inputs[3]]

            with vs.variable_scope("one") as one_scope:
                output_one, _ = tf.contrib.rnn.static_rnn(
                    cell=cell_one,
                    inputs=one_inputs,
                    dtype=tf.float32,
                    scope=one_scope)

            with vs.variable_scope("two") as two_scope:
                output_two, _ = tf.contrib.rnn.static_rnn(
                    cell=cell_two,
                    inputs=two_inputs,
                    dtype=tf.float32,
                    scope=two_scope)

            with vs.variable_scope("three") as three_scope:
                output_three, _ = tf.contrib.rnn.static_rnn(
                    cell=cell_three,
                    inputs=three_inputs,
                    dtype=tf.float32,
                    scope=three_scope)

            with vs.variable_scope("four") as four_scope:
                output_four, _ = tf.contrib.rnn.static_rnn(
                    cell=cell_four,
                    inputs=four_inputs,
                    dtype=tf.float32,
                    scope=four_scope)

            with vs.variable_scope("five") as five_scope:
                output_five, _ = tf.contrib.rnn.static_rnn(
                    cell=cell_five,
                    inputs=five_inputs,
                    dtype=tf.float32,
                    scope=five_scope)

        final_output_one = [output_one[4], output_one[0], output_one[1], output_one[2], output_one[3]]
        final_output_two = [output_two[0], output_two[4], output_two[1], output_two[2], output_two[3]]
        final_output_three = [output_three[0], output_three[1], output_three[4], output_three[2], output_three[3]]
        final_output_four = [output_four[0], output_four[1], output_four[2], output_four[4], output_four[3]]

        flat_outputs = tuple(array_ops.concat([one, two, three, four, five], 1)
                             for one, two, three, four, five in
                             zip(final_output_one, final_output_two, final_output_three, final_output_four,
                                 output_five))

        outputs = nest.pack_sequence_as(
            structure=output_one, flat_sequence=flat_outputs)

        return outputs

    def base_build_critic_network(self, state):
        outputs = self.shared_dense_layer("critic_layer1", state, CRITIC_LAYER1)
        outputs = tf.unstack(outputs, NUM_AGENTS, 1)

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
        # Build communication net
        outputs = self.critic_comm(lstm_cell_one, lstm_cell_two, lstm_cell_three, lstm_cell_four, lstm_cell_five,
                                   outputs)
        outputs = tf.stack(outputs, 1)
        outputs = self.shared_dense_layer("critic_layer2", outputs, CRITIC_OUTPUT_LEN)

        return outputs

    def base_build_actor_network(self, observation, input_state):
        outputs = self.shared_dense_layer("actor_layer1", observation, ACTOR_LAYER1)
        outputs = tf.unstack(outputs, NUM_AGENTS, 1)
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
        outputs, output_state = self.actor_comm(lstm_cell_one, lstm_cell_two, lstm_cell_three, lstm_cell_four,
                                                lstm_cell_five, lstm_memory_cell, input_state,
                                                outputs)

        outputs = tf.stack(outputs, 1)

        outputs = self.shared_dense_layer("actor_layer2", outputs, ACTOR_OUTPUT_LEN)

        return outputs, output_state

    def actor_build_network(self, name, observation, input_state):
        """Building a multi-agent actor net"""
        with tf.variable_scope(name):
            outputs, output_state = self.base_build_actor_network(observation, input_state)
            action_probs = tf.nn.softmax(outputs, axis=-1)
            log_action_probs = tf.nn.log_softmax(outputs, axis=-1)
            # Policy without noise
            mu = tf.argmax(outputs, axis=-1)
            # Policy with noise
            policy_dist = tf.distributions.Categorical(logits=outputs)
            pi = policy_dist.sample()
            return mu, pi, action_probs, log_action_probs, output_state

    @staticmethod
    def shared_dense_layer(name, observation, output_len):
        """The weights of dense layer are shared."""
        all_outputs = []
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for j in range(NUM_AGENTS):
                agent_obs = observation[:, j]
                outputs = tf.layers.dense(agent_obs, output_len, name="dense")
                all_outputs.append(outputs)
            all_outputs = tf.stack(all_outputs, 1)
        return all_outputs

    def critic_build_network(self, name, state, action):
        """Building a multi-agent critic net"""
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            outputs = self.base_build_critic_network(state)
            # Responsible output
            q_a = tf.reduce_sum(tf.multiply(outputs, action), axis=-1)
            return outputs, q_a
