import tensorflow as tf



TAU = 0.001

class DDPG:
    def __init__(self, env, name):
        self.env = env
        self.name = name

        self.state_dim = env.observation_space
        self.action_dim = env.action_space.shape[0]
        self.action_bound = [env.action_space.low, env.action_spcae.high]

        with tf.variable_scope('placeholder'):
            self.actions = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.action_dim))
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None])

            # Critic 의 update를 위해서 사용 될 estimated value 를 받아올 ph
            self.returns = tf.placeholder(dtype=tf.float32, shape=[None])

            # Actor 의 update를 위해서 Critic의 gradient를 받아올 ph
            self.critic_gradients = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.action_dim))

        with tf.variable_scope('main_network'):
            self.action, self.scaled_action, self.observations = self.actor_network('main')
            self.pred_v, self.c_observations = self.critic_network('main')
            self.actor_vars, self.critic_vars = self.get_variables()

        with tf.variable_scope('target_network'):
            self.target_action, self.target_scaled_action, self.target_observations = self.actor_network('target')
            self.target_pred_v, self.target_c_observations = self.critic_network('target')
            self.target_actor_vars, self.target_critic_vars = self.get_variables()


        with tf.variable_scope('critic_update'):
            # R(t) - V(S_(t))
            # ------------------------- :: returns
            #                              _______ :: pred_v
            self.loss_c = tf.reduce_mean(tf.squared_difference(self.returns, self.pred_v))

        with tf.variable_scope('actor_update'):
            self.actor_gradients = tf.gradients(self.scaled_action, self.actor_vars, -self.critic_gradients)
            self.grads_and_vars = list(zip(self.actor_gradients, self.actor_vars))
            self.train_op = tf.train.AdamOptimizer(0.00025).apply_gradients(self.grads_and_vars)

        with tf.variable_scope('actor_assign'):
            self.assign_ops = [ self.target_actor_vars.assign(tf.multiply(self.actor_vars[i], TAU) + tf.multiply(self.target_actor_vars[i], 1- TAU)) for i in range(len(self.actor_vars)) ]

    #====================================Actor Function===============================================================================

    def train_actor(self, obs, critic_grads):
        return tf.get_default_session().run(self.train_op, feed_dict={self.observations : obs, self.critic_gradients : critic_grads})

    def predict_actor(self, obs):
        return tf.get_default_session().run(self.scaled_action, feed_dict={self.observations : obs})

    def predict_target_actor(self, obs):
        return tf.get_default_session().run(self.target_scaled_action, feed_dict={self.target_observations : obs})

    def soft_update_actor(self):
        tf.get_default_session().run(self.assign_ops)

    # ====================================Actor Function End==========================================================================

    # ====================================Critic Function===============================================================================




    # ====================================Critic Function End==========================================================================

    def actor_network(self, name):

        observations = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.state_dim))

        with tf.variable_scope(name):
            with tf.variable_scope('actor'):
                h1 = tf.layers.dense(self.observations, units=200, activation=tf.nn.relu6)
                h2 = tf.layers.dense(h1, units=50, activation=tf.nn.relu6)
                action = tf.layers.dense(h2, units=self.action_dim, activation=tf.nn.sigmoid)
                scaled_action =  tf.multiply(action, self.action_bound[1])
                self.actor_scope = tf.get_variable_scope().name
        return action, scaled_action, observations

    def critic_network(self, name):

        observations = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.state_dim))

        with tf.variable_scope(name):
            with tf.variable_scope('critic'):
                h1 = tf.layers.dense(self.observations, units=100, activation=tf.nn.relu6)
                v = tf.layers.dense(h1, units=1, activation=None)
                self.critic_scope = tf.get_variable_scope().name
        return v, observations



    #===============================Common Function==========================================================

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.actor_scope), tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.critic_scope)




