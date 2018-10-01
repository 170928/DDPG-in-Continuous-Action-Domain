import tensorflow as tf

class A2C:
    def __init__(self, env, name):
        self.env = env
        self.name = name

        self.state_dim = env.observation_space
        self.action_dim = env.action_space.shape[0]
        self.action_bound = [env.action_space.low, env.action_spcae.high]

        with tf.variable_scope('placeholder'):
            self.states = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.state_dim))
            self.actions = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.action_dim))
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None])

            # Critic 의 update를 위해서 사용 될 estimated value 를 받아올 ph
            self.target_v = tf.placeholder(dtype=tf.float32, shape=[None])

        with tf.varialbe_scope('network'):
            self.mu_due1, self.sigma_due2, self.mu_due2, self.sigma_due2 = self.actor_network()
            self.pred_v = self.critic_network()
            self.actor_vars, self.critic_vars = self.get_variables()

        with tf.variable_scope('advantage'):
            # r(t) + gamma * V(S_(t+1)) - V(S_(t))
            # ------------------------- :: target_v
            #                              _______ :: pred_v
            adv = tf.subtract(self.target_v, self.pred_v)

        with tf.variable_scope('critic_loss'):

        with tf.variable_scope('actor_loss'):



    def actor_network(self):
        with tf.variable_scope('actor'):
            h1 = tf.layers.dense(self.states, units=200, activation=tf.nn.relu6)
            h2 = tf.layers.dense(h1, units=50, activation=tf.nn.relu6)
            mu1 = tf.layers.dense(h2, units=self.action_dim, activation=tf.nn.tanh)
            sigma1 = tf.layers.dense(h2, units=self.action_dim, activation=tf.nn.softplus)

            mu2 = tf.layers.dense(h2, units=self.action_dim, activation=tf.nn.tanh)
            sigma2 = tf.layers.dense(h2, units=self.action_dim, activation=tf.nn.softplus)

            self.actor_scope = tf.get_variable_scope().name
        return mu1, sigma1, mu2, sigma2

    def critic_network(self):
        with tf.variable_scope('critic'):
            h1 = tf.layers.dense(self.states, units=100, activation=tf.nn.relu6)
            v = tf.layers.dense(h1, units=1, activation=None)
            self.critic_scope = tf.get_variable_scope().name
        return v

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.actor_scope), tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.critic_scope)




