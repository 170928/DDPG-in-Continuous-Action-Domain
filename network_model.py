import tensorflow as tf



TAU = 0.001

class DDPG:
    def __init__(self, env):
        self.env = env

        self.state_dim = env.observation_space
        self.action_dim = env.action_space.shape[0]
        self.action_bound = [env.action_space.low, env.action_spcae.high]

        with tf.variable_scope('placeholder'):
            # Critic 의 update를 위해서 사용
            self.pred_policy = tf.placeholder(tf.float32, [None, 1])

            # Actor 의 update를 위해서 Critic의 gradient를 받아올 ph
            self.critic_gradients = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.action_dim))

        with tf.variable_scope('main_network'):
            self.action, self.scaled_action, self.observations, self.actor_scope = self.actor_network('main')
            self.pred_v, self.c_observations, self.actions, self.critic_scope = self.critic_network('main')
            self.actor_vars, self.critic_vars = self.get_variables(self.actor_scope, self.critic_scope)

        with tf.variable_scope('target_network'):
            self.target_action, self.target_scaled_action, self.target_observations, self.target_actor_scope = self.actor_network('target')
            self.target_pred_v, self.target_c_observations, self.target_actions, self.target_critic_scope = self.critic_network('target')
            self.target_actor_vars, self.target_critic_vars = self.get_variables(self.target_actor_scope, self.target_critic_scope)


        with tf.variable_scope('actor_update'):
            self.actor_gradients = tf.gradients(self.scaled_action, self.actor_vars, -self.critic_gradients)
            self.grads_and_vars = list(zip(self.actor_gradients, self.actor_vars))
            self.train_op_actor = tf.train.AdamOptimizer(0.00025).apply_gradients(self.grads_and_vars)

        with tf.variable_scope('critic_update'):
            self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.pred_policy, self.pred_v))))
            self.train_op_critic = tf.train.AdamOptimizer(0.002).minimize(self.loss)
            self.critic_grads = tf.gradients(self.pred_v, self.action)

        with tf.variable_scope('actor_assign'):
            self.assign_ops_actor = [ self.target_actor_vars[i].assign(tf.multiply(self.actor_vars[i], TAU) + tf.multiply(self.target_actor_vars[i], 1 - TAU)) for i in range(len(self.actor_vars)) ]

        with tf.variable_scope('critic_assign'):
            self.assign_ops_critic = [ self.target_critic_vars[i].assign(tf.multiply(self.critic_vars[i], TAU) + tf.multiply(self.target_critic_vars[i], 1 - TAU)) for i in range(len(self.critic_vars)) ]
    #====================================Actor Function===============================================================================

    def train_actor(self, obs, critic_grads):
        return tf.get_default_session().run(self.train_op_actor, feed_dict={self.observations : obs, self.critic_gradients : critic_grads})

    def predict_actor(self, obs):
        return tf.get_default_session().run(self.scaled_action, feed_dict={self.observations : obs})

    def predict_target_actor(self, obs):
        return tf.get_default_session().run(self.target_scaled_action, feed_dict={self.target_observations : obs})

    def soft_update_actor(self):
        return tf.get_default_session().run(self.assign_ops_actor)

    # ====================================Actor Function End==========================================================================

    # ====================================Critic Function===============================================================================

    def train_critic(self, obs, actions, pred_policy):
        return tf.get_default_session().run(self.train_op_critic, feed_dict={self.c_observations : obs, self.actions : actions, self.pred_policy : pred_policy})

    def predict_critic(self, obs, actions):
        return tf.get_default_session().run(self.pred_v, feed_dict={self.c_observations : obs, self.actions : actions})

    def predict_target_critic(self, obs, actions):
        return tf.get_default_session().run(self.target_pred_v, feed_dict={self.target_c_observations : obs, self.target_actions : actions})

    def critic_gradients(self, obs, actions):
        return tf.get_default_session().run(self.critic_grads, feed_dict={
            self.c_observations: obs,
            self.actions: actions
        })

    def sotf_update_critic(self):
        return tf.get_default_session().run(self.assign_ops_critic)

    # ====================================Critic Function End==========================================================================

    def actor_network(self, name):

        observations = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.state_dim))

        with tf.variable_scope(name):
            with tf.variable_scope('actor'):
                h1 = tf.layers.dense(self.observations, units=200, activation=tf.nn.relu6)
                h2 = tf.layers.dense(h1, units=50, activation=tf.nn.relu6)
                action = tf.layers.dense(h2, units=self.action_dim, activation=tf.nn.sigmoid)
                scaled_action =  tf.multiply(action, self.action_bound[1])
                actor_scope = tf.get_variable_scope().name

        return action, scaled_action, observations, actor_scope

    def critic_network(self, name):

        observations = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.state_dim))
        actions = tf.placeholder(tf.float32, [None] + list(self.action_dim))

        with tf.variable_scope(name):
            with tf.variable_scope('critic'):
                h1 = tf.layers.dense(self.observations, units=100, activation=tf.nn.relu6)
                v = tf.layers.dense(h1, units=1, activation=None)
                critic_scope = tf.get_variable_scope().name
        return v, observations, actions, critic_scope

    #===============================Common Function==========================================================

    def get_variables(self, actor_scope, critic_scope):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=actor_scope), tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=critic_scope)




