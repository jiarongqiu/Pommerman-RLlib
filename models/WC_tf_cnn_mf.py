from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf
from ray.rllib.models.tf.misc import normc_initializer

tf = try_import_tf()


class TFCNN(TFModelV2):
    """Generic vision network implemented in ModelV2 API."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(TFCNN, self).__init__(obs_space, action_space,
                                    num_outputs, model_config, name)
        self.num_outputs = num_outputs
        self.init()

    def init(self):
        _board = tf.keras.layers.Input(shape=[11, 11, 2], name="board")
        _attribute = tf.keras.layers.Input(shape=[2], name="attribute")

        net = _board
        net = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same", activation=tf.nn.relu,
                                     kernel_initializer=normc_initializer(0.01))(net)
        net = tf.keras.layers.BatchNormalization()(net)

        net = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same", activation=tf.nn.relu,
                                     kernel_initializer=normc_initializer(0.01))(net)
        net = tf.keras.layers.BatchNormalization()(net)

        net = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same", activation=tf.nn.relu,
                                     kernel_initializer=normc_initializer(0.01))(net)
        net = tf.keras.layers.BatchNormalization()(net)

        net = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same", activation=tf.nn.relu,
                                     kernel_initializer=normc_initializer(0.01))(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.reshape(net, (-1, net.shape[-1]*net.shape[-2]*net.shape[-3]))

        # net = tf.concat([net, _attribute], axis=1)

        net = tf.keras.layers.Dense(1024, activation=tf.nn.relu, kernel_initializer=normc_initializer(0.01))(net)
        # net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Dense(1024, activation=tf.nn.relu, kernel_initializer=normc_initializer(0.01))(net)


        action_out = tf.keras.layers.Dense(self.num_outputs)(net)
        value_out = tf.keras.layers.Dense(1)(net)

        self.base_model = tf.keras.Model([_board, _attribute], [action_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        board = input_dict['obs']['board']
        danger = input_dict['obs']['danger']
        bomb_blast_strength = input_dict['obs']['bomb_blast_strength']
        bomb_life = input_dict['obs']['bomb_life']
        flame_life = input_dict['obs']['flame_life']
        position = input_dict['obs']['position']
        ammo = input_dict['obs']['ammo']
        blast_strength = input_dict['obs']['blast_strength']

        _board = tf.cast(tf.reshape(tf.stack([board, danger]), (-1, 11, 11, 2)),
                         tf.float32)
        _attributes = tf.cast(tf.reshape(tf.concat([ammo, blast_strength], axis=1), (-1, 2)), tf.float32)

        action_out, self.value_out = self.base_model((_board, _attributes))
        return action_out, state

    def value_function(self):
        return tf.reshape(self.value_out, [-1])
