import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

GAMMA = 0.95
MEMORY_SIZE=1000000
BATCH_SIZE=20
EXPLORATION_MAX=1.0
EXPLORATION_MIN=0.01
EXPLORATION_DECAY=0.995
MAX_STEPS = 1000


class DQN:
    
    def __init__(self,observation_space,action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.exploration_rate = EXPLORATION_MAX
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = Sequential([Dense(24,input_shape=(observation_space,),activation='relu'),
                                Dense(24,activation='relu'),
                                Dense(self.action_space,activation='linear')
                                ])
        self.model.compile(loss='mse', optimizer=Adam(lr=0.001))
        
    def act(self,state):
        if np.random.rand()<self.exploration_rate:
            return random.randrange(self.action_space)
        q_values=self.model.predict(state)
        return np.argmax(q_values[0])
    def remember(self,state,action,reward,next_state,terminal):
        self.memory.append([state,action,reward,next_state,terminal])
    
    def experiance_replay(self):
        if len(self.memory)<BATCH_SIZE:
            return
        batch = random.sample(self.memory,BATCH_SIZE)
        for state,action,reward,next_state,terminal in batch:
            if terminal:
                q_update = reward
            if not terminal:
                q_update = reward + GAMMA*np.amax(self.model.predict(next_state)[0])
                q_target = self.model.predict(state)
                q_target[0][action]=q_update
                self.model.fit(state,q_target,verbose=0)
        self.exploration_rate = max(EXPLORATION_MIN, EXPLORATION_DECAY*self.exploration_rate)
       
def cartpole():
    env=gym.make('CartPole-v1')
    observation_space=env.observation_space.shape[0]
    action_space=env.action_space.n
    dqn_solver=DQN(observation_space,action_space)
    
    episode = 0
    while True:
        state = env.reset()
        state = np.reshape(state,[1,observation_space])
        total_rewards = 0
        steps = 0
        
        for i in range(MAX_STEPS):
            '''env.render()'''
            action = dqn_solver.act(state)
            next_state,reward,terminal,info = env.step(action)
            reward = reward if not terminal else -reward
            next_state = np.reshape(next_state,[1,observation_space])
            dqn_solver.remember(state,action,reward,next_state,terminal)
            dqn_solver.experiance_replay()
            state = next_state
            total_rewards += reward
            steps+=1
            if terminal:
                break
        avg_reward = total_rewards/steps
        episode+=1
        print('total_reward for episode:{0} = {1}, total_steps = {2}'.format(episode,total_rewards,steps))
            
            
            
if __name__=='__main__':
    cartpole()
        
