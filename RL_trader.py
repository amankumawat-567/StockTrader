import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from Tensorflow.keras.optimizers import Adam

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler

def get_data():
    # return a T x D sequence
    # T is time steps and D is number of stocks
    # [Apple, Motorola, Starbucks]
    df = pd.read_csv('Dataset\\aapl_msi_sbux.csv')
    return df.values

class ReplayBuffer:
    # Experience replay memory
    
    def __init__(self, s_dim, a_dim, size = 32):
        self.s_buf = np.zeros([size, s_dim], dtype=np.float32)
        self.next_s_buf = np.zeros([size, s_dim], dtype=np.float32)
        self.a_buf = np.zeros(size, dtype=np.uint8)
        self.r_buf = np.zeros(size, dtype=np.float32)
        self.done_buff = np.zeros(size, dtype=np.uint8)
        self.ptr, self.size, self.max_size = 0, 0, size
        
    def store(self, s, next_s, a, r, done):
        self.s_buf[self.ptr] = s
        self.next_s_buf[self.ptr] = next_s
        self.a_buf[self.ptr] = a
        self.r_buf[self.ptr] = r
        self.done_buff[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample_batch(self, batch_size = 32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            s=self.s_buf[idxs],
            next_s=self.next_s_buf[idxs],
            a=self.a_buf[idxs],
            r=self.r_buf[idxs],
            done=self.done_buff[idxs]
            )
        
def get_scaler(env):
    # It returns a StandardScaler object to scale the states
    
    states = []
    for _ in range(env.num_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break
        
    scaler = StandardScaler()
    scaler.fit(states)
    return scaler

def check_make_dir(directory):
    # Check whether directory exists or not and make it if not
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def MLP(input_dim, num_action, num_hidden_layers = 1, hidden_dim = 32):
    """ A Multi-Layer Perceptron"""
    
    # input layer
    i = Input(shape=(input_dim,))
    x = i
    
    # hidden layers
    for _ in range(num_hidden_layers):
        x = Dense(hidden_dim, activation='relu')(x)
        
    # output layer    
    x = Dense(num_action)(x)
    
    # creating model
    model = Model(i, x)
    model.compile(loss='mse',
                  optimizer='adam')
    print((model.summary()))
    return model
    
class MultiStockEnv:
    """
    A 3 Stock trading environment
    State : vector of size 7 (num_stocks * 2 + 1)
        - Shares of Stokes owned
        - Prices of Stokes
        - Cash owned
    Action : Categorical variable with 27 (3^3) possibilities
        - for eack stock you can:
            sell -> 0
            hold -> 1
            buy  -> 2
    """
    def __init__(self, data, initial_investment=20000):
        
        # data
        self.stock_price_history = data
        self.num_step, self.num_stock = self.stock_price_history.shape
        
        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None
        
        # Actions
        self.action_space = np.arange(3**self.num_stock)
        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.num_stock)))
        
        # State dimensions
        self.state_dim = self.num_stock * 2 + 1
        
        self.reset()
        
    def reset(self):
        """
        Reset the environment
        """
        self.cur_step = 0
        self.stock_owned = np.zeros(self.num_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        return self._get_s()
    
    def step(self, action):
        assert action in self.action_space
        
        # get current value before performing the action
        prev_val = self._get_val()
        
        # update price, i.e. go to the next day
        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step]
        
        # perform the trade
        self._trade(action)
        
        # get the new value after taking the action
        cur_val = self._get_val()
        
        reward = cur_val - prev_val
        done = (self.cur_step == (self.num_step - 1))
        info = {'cur_val' : cur_val}
        
        return self._get_s(), reward, done, info
    
    def _get_s(self):
        """
        Returns the current state vector
        """
        s = np.empty(self.state_dim)
        s[:self.num_stock] = self.stock_owned
        s[self.num_stock: 2*self.num_stock] = self.stock_price
        s[-1] = self.cash_in_hand
        return s
    
    def _get_val(self):
        """
        Returns the current value of portfolio
        """
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand
    
    def _trade(self, action):
        """
        index the action we want to perform
        
        ### Important 
         - This section is can be customized according to you stock broker
           API and your oun Technique to buy and sell
         - For now its not linked to any api and its based on Greedy Technique
           to buy and sell stocks
        """
        action_vec = self.action_list[action]
        
        # determine which stocks to buy or sell
        sell_index = [] # stores index of stocks we want to sell
        buy_index = [] # stores index of stocks we want to buy
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)

        # sell any stocks we want to sell
        # then buy any stocks we want to buy
        if sell_index:
            # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
            for i in sell_index:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0
        if buy_index:
            # NOTE: when buying, we will loop through each stock we want to buy,
            #       and buy one share at a time until we run out of cash
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] += 1 # buy one share
                        self.cash_in_hand -= self.stock_price[i]
                    else:
                        can_buy = False
                        
class DQNAgent(object):
    """
    This is the Agent that will do the trade for us
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(state_size, action_size, size=500)
        self.gamma = 0.95 # discount rate
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = MLP(state_size,action_size)
        
    def update_replay_memory(self,s,next_s,a,r,done):
        self.memory.store(s,next_s,a,r,done)
        
    def act(self, s):
        """
        Choose which action to perform
        ### Epsilon-Greedy
        """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(s)
        return np.argmax(act_values[0]) # return action
    
    def replay(self, batch_size=32):
        # check if memory has enought samples
        if self.memory.size < batch_size:
            return
        
        minibatch = self.memory.sample_batch(batch_size)
        states = minibatch['s']
        actions = minibatch['a']
        rewards = minibatch['r']
        done = minibatch['done']
        next_states = minibatch['next_s']
        
        target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)
        
        # The value of terminal states is zero
        # so set the target to be the reward only
        target[done] = rewards[done]
        # there is no as such terminal state in stocks its basically end of data
        
        # With the Keras API, the target (usually) must have the same
        # shape as the predictions.
        # However, we only need to update the network for the actions
        # which were actually taken.
        # We can accomplish this by setting the target to be equal to
        # the prediction for all values.
        # Then, only change the targets for the actions taken.
        # Q(s,a)
        target_full = self.model.predict(states)
        target_full[np.arange(batch_size), actions] = target
        
        self.model.train_on_batch(states, target_full)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def load(self, name):
        self.model.load_weights(name)
        
    def save(self, name):
        self.model.save_weights(name)
        

def play_one_episode(agent, env, is_train):
    
    state = env.reset()
    state = scaler.transform([state])
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])
        if is_train == 'train':
            agent.update_replay_memory(state, next_state, action, reward, done)
            agent.replay(batch_size)
        state = next_state
        
    return info['cur_val']

if __name__ == '__main__':
    
    # config
    models_folder = 'Models'
    rewards_folder = 'Rewards'
    num_episodes = 2000
    batch_size = 32
    initial_investment = 20000
    
    # parser object help us to run our code through command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='either "train" or "test"')
    args = parser.parse_args()
    
    check_make_dir(models_folder)
    check_make_dir(rewards_folder)
    
    data = get_data()
    num_time_steps, num_stocks = data.shape
    
    n_train = num_time_steps // 2
    
    train_data = data[:n_train]
    test_data = data[n_train:]
    
    env = MultiStockEnv(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)
    
    # store the final value of portfolio (end of episode)
    portfolio_value = []
    
    if args.mode == 'test':
        # load previous scaler
        with open(f'{models_folder}/scaler.pkl','rb') as f:
            scaler = pickle.load(f)
            
        env = MultiStockEnv(test_data, initial_investment)
        
        agent.epsilon = 0.01
        
        agent.load(f'{models_folder}/dqn.h5')
        
    for e in range(num_episodes):
        t0 = datetime.now()
        val = play_one_episode(agent, env, args.mode)
        dt = datetime.now() - t0
        print(f'Episode {e+1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}')
        portfolio_value.append(val)
        
    if args.mode == 'train':
        # save the DQN
        agent.save(f'{models_folder}/dqn.h5')
        
        # save Scaler
        with open(f'{models_folder}/scaler.pkl','wb') as f:
            pickle.dump(scaler, f)
    
    # save portfolio values        
    np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)