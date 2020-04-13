from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


class TFCNN(TFModelV2):
    """Generic vision network implemented in ModelV2 API."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(TFCNN, self).__init__(obs_space, action_space,
                                            num_outputs, model_config, name)
        self.num_outputs = num_outputs
        self.init2()

    def init(self):
        kernel_initializer = 'glorot_uniform'
        # kernel_initializer = 'glorot_normal'
        _board = tf.keras.layers.Input(shape=[11,11,4], name="board")
        _attribute = tf.keras.layers.Input(shape=[4],name="attribute")

        net = _board
        net = tf.keras.layers.Conv2D(256,3,1,kernel_initializer=kernel_initializer,padding="same",activation=tf.nn.relu)(net)
        net = tf.keras.layers.Conv2D(256, 3, 1,kernel_initializer=kernel_initializer, padding="same", activation=tf.nn.relu)(net)
        net = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(net)

        net = tf.keras.layers.Conv2D(128, 3, 1,kernel_initializer=kernel_initializer, padding="same", activation=tf.nn.relu)(net)
        net = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(net)
        net = tf.keras.layers.Conv2D(128, 3, 1,kernel_initializer=kernel_initializer, padding="same", activation=tf.nn.relu)(net)
        net = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(net)
        net = tf.reshape(net,(-1,net.shape[-1]))

        net = tf.concat([net,_attribute],axis=1)

        net = tf.keras.layers.Dense(64,kernel_initializer=kernel_initializer,activation=tf.nn.relu)(net)

        action_out = tf.keras.layers.Dense(self.num_outputs,kernel_initializer=kernel_initializer)(net)
        value_out = tf.keras.layers.Dense(1,kernel_initializer=kernel_initializer)(net)

        self.base_model = tf.keras.Model([_board,_attribute],[action_out,value_out])
        self.register_variables(self.base_model.variables)


    def init2(self):

        _board = tf.keras.layers.Input(shape=[11,11,4], name="board")
        _attribute = tf.keras.layers.Input(shape=[4],name="attribute")

        net = _board
        net = tf.keras.layers.Conv2D(256,3,1,padding="valid",activation=tf.nn.relu)(net) # 9x9
        net = tf.keras.layers.Conv2D(256,3,1, padding="valid", activation=tf.nn.relu)(net) # 7x7

        net = tf.keras.layers.Conv2D(128, 3, 1, padding="valid", activation=tf.nn.relu)(net) # 5x5
        net = tf.keras.layers.Conv2D(128, 3, 1, padding="valid", activation=tf.nn.relu)(net) # 3x3

        net = tf.keras.layers.Conv2D(64, 3, 1, padding="valid", activation=tf.nn.relu)(net)  # 1x1
        net = tf.reshape(net,(-1,net.shape[-1]))

        net = tf.concat([net,_attribute],axis=1)

        action_net = tf.keras.layers.Dense(64,activation=tf.nn.relu)(net)
        action_out = tf.keras.layers.Dense(self.num_outputs)(action_net)

        value_net = tf.keras.layers.Dense(64, activation=tf.nn.relu)(net)
        value_out = 10*tf.keras.layers.Dense(1)(value_net)

        self.base_model = tf.keras.Model([_board,_attribute],[action_out,value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        board = input_dict['obs']['board']
        bomb_blast_strength = input_dict['obs']['bomb_blast_strength']
        bomb_life = input_dict['obs']['bomb_life']
        flame_life = input_dict['obs']['flame_life']
        position = input_dict['obs']['position']
        ammo = input_dict['obs']['ammo']
        blast_strength = input_dict['obs']['blast_strength']

        _board = tf.cast(tf.reshape(tf.stack([board, bomb_blast_strength, bomb_life, flame_life]),(-1,11,11,4)),tf.float32)
        _attributes = tf.cast(tf.reshape(tf.concat([position, ammo, blast_strength], axis=1),(-1,4)),tf.float32)

        # normalization
        _board = _board/13.0*2.0-1.0            # [-1,1]
        _attributes = _attributes/13.0          # [0, ~1


        action_out,self.value_out = self.base_model((_board,_attributes))
        return action_out,state

    def value_function(self):
        return tf.reshape(self.value_out, [-1])