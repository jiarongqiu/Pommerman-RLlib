from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


class TFCNN2(TFModelV2):
    """Generic vision network implemented in ModelV2 API."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(TFCNN2, self).__init__(obs_space, action_space,
                                            num_outputs, model_config, name)
        self.num_outputs = num_outputs
        self.init()

    def init(self):

        _board = tf.keras.layers.Input(shape=[11,11,10], name="board")
        net = _board
        net = tf.keras.layers.Conv2D(256,3,1,padding="valid",activation=tf.nn.relu)(net) # 9x9
        net = tf.keras.layers.Conv2D(256,3,1, padding="valid", activation=tf.nn.relu)(net) # 7x7

        net = tf.keras.layers.Conv2D(128, 3, 1, padding="valid", activation=tf.nn.relu)(net) # 5x5
        net = tf.keras.layers.Conv2D(128, 3, 1, padding="valid", activation=tf.nn.relu)(net) # 3x3

        net = tf.keras.layers.Conv2D(64, 3, 1, padding="valid", activation=tf.nn.relu)(net)  # 1x1
        net = tf.reshape(net,(-1,net.shape[-1]))

        action_net = tf.keras.layers.Dense(64,activation=tf.nn.relu)(net)
        action_out = tf.keras.layers.Dense(self.num_outputs)(action_net)

        value_net = tf.keras.layers.Dense(64, activation=tf.nn.relu)(net)
        value_out = tf.keras.layers.Dense(1)(value_net)

        self.base_model = tf.keras.Model(_board,[action_out,value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        board = input_dict['obs']
        _board = tf.cast(board,tf.float32)

        action_out,self.value_out = self.base_model(_board)
        return action_out,state

    def value_function(self):
        return tf.reshape(self.value_out, [-1])