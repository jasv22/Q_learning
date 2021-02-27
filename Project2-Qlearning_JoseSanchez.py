#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[426]:


class Environment:
    
    def __init__(self,COLUMNS = 3,ROWS = 3,START = (0,0) ,GOAL = (2,2),OBSTACLES = []):
        self.rows = ROWS
        self.columns = COLUMNS
        self.board = np.zeros([ROWS,COLUMNS])
        self.start = START
        self.position = self.start
        self.restart = False
        self.goal = GOAL
        self.board[GOAL] = 1
        self.obstacles = OBSTACLES
        for i in self.obstacles:
            self.board[i] = -1        
    
    def reward(self):
        return self.board[self.position]
    
    def position_check(self):
        if(self.position == self.goal or self.position in self.obstacles):
            self.restart = True
            #print("go back")
        #else:
            #print("continue")
    
    def chooseAction(self,action):
        if action == "up":
            return np.random.choice(["up", "left", "right"], p=[0.8, 0.1, 0.1])
        if action == "down":
            return np.random.choice(["down", "left", "right"], p=[0.8, 0.1, 0.1])
        if action == "left":
            return np.random.choice(["left", "up", "down"], p=[0.8, 0.1, 0.1])
        if action == "right":
            return np.random.choice(["right", "up", "down"], p=[0.8, 0.1, 0.1])
 
    def nxtPosition(self, action):
        #nxtAction = self.chooseAction(action)
        nxtAction = action
        nxtPosition = self.position
        if nxtAction == "up":
            nxtPosition = (self.position[0] - 1, self.position[1])
        elif nxtAction == "down":
            nxtPosition = (self.position[0] + 1, self.position[1])
        elif nxtAction == "left":
            nxtPosition = (self.position[0], self.position[1] - 1)
        else: #right
            nxtPosition = (self.position[0], self.position[1] + 1)
            
        #print(nxtPosition[0],nxtPosition[1])
        
        if nxtPosition[0] < 0 or nxtPosition[0] >= self.rows:
            #print("rows")
            return self.nxtPosition(action)
        elif nxtPosition[1] < 0 or nxtPosition[1] >= self.columns:
            #print("columns")
            return self.nxtPosition(action)
        else:
            self.position = nxtPosition
            return nxtAction      
    
    def showBoard(self):
        
        for i in range(0, self.rows):
            print('-----'* self.rows)
            out = ' | '
            for j in range(0, self.columns):
                if self.board[i, j] == 1:
                    token = 'G'
                if self.board[i, j] == -1:
                    token = 'Z'
                if self.board[i, j] == 0:
                    token = '0'
                if (i,j) == self.start:
                    token = 'S'
                if (i,j) == self.position:
                    token = 'X'
                out += token + ' | '
            print(out)
        print('-----' * self.rows)


# In[435]:


class Agent:
    def __init__(self,BOARD = Environment(), LR = 0.2,EXP_R = 0.3,GAMMA = 0.9):
        self.states = []  # record position and action taken at the position
        self.actions = ["up","down","left","right"]
        self.action = ""
        self.board = BOARD
        self.restart = self.board.restart
        self.lr = LR
        self.exp_rate = EXP_R
        self.decay_gamma = GAMMA
        self.Q_values = {}
        for i in range(self.board.rows):
            for j in range(self.board.columns):
                self.Q_values[(i,j)] = {}
                for a in self.actions:
                    self.Q_values[i,j][a]=0
    
    def move(self,action):
        position = self.board.nxtPosition(action)
        return position
        
    def reset(self):
        self.states = []
        self.board.position = self.board.start
        self.restart = False
        self.board.restart = False
    
    def showQ_values(self):
        for i in self.Q_values:
            print(i,self.Q_values[i])
    
    def actionvalid(self,action):
        nxtPosition = self.board.position
        if action == "up":
            nxtPosition = (self.board.position[0] - 1, self.board.position[1])
        elif action == "down":
            nxtPosition = (self.board.position[0] + 1, self.board.position[1])
        elif action == "left":
            nxtPosition = (self.board.position[0], self.board.position[1] - 1)
        else: #right
            nxtPosition = (self.board.position[0], self.board.position[1] + 1)
        
        if nxtPosition[0] < 0 or nxtPosition[0] >= self.board.rows:
            #print("rows")
            return False
        elif nxtPosition[1] < 0 or nxtPosition[1] >= self.board.columns:
            #print("columns")
            return False
        else:
            return True
    
    def get_actions():
        return self.actions
    
    def chooseAction(self, ActionsRemaining):
        # choose action with most expected value
        action = ""
        actions_remaning = ActionsRemaining
        mx_nxt_reward = self.Q_values[self.board.position][np.random.choice(actions_remaning)]
        #print(actions_remaning)
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(actions_remaning)
            #print("random - :",action)
        else:
            # greedy action
            for a in actions_remaning:
                current_position = self.board.position
                nxt_reward = self.Q_values[current_position][a]
                #print("nxt R: ",nxt_reward)
                #print("MX nxt R: ",mx_nxt_reward)
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
            #print("greedy - :",action)
        
        actions_remaning.remove(action)
        #print("current pos: {}, greedy action: {}".format(self.board.position, action))
        if self.actionvalid(action):
            return action
        else:
            return self.chooseAction(actions_remaning)
    
    def Q_learning(self,rounds=10):
        i = 0
        
        print("Initial Q Values: ")
        self.showQ_values()
        print("--- Starting ---")
        self.board.showBoard()
        while i < rounds:
            
            if self.board.restart:
                # back propagate
                reward = self.board.reward()
                for a in self.actions:
                    self.Q_values[self.board.position][a] = reward
                print("Game End Reward", reward)
                for s in reversed(self.states):
                    current_q_value = self.Q_values[s[0]][s[1]]
                    reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)
                    self.Q_values[s[0]][s[1]] = round(reward, 3)
                self.reset()
                i += 1
            else:
                action = self.chooseAction(["up","down","left","right"])
                # append trace
                self.states.append([(self.board.position), action])
                print("current position {} action {}".format(self.board.position, action))
                action = self.move(action)
                # by taking the action, it reaches the next state
                # mark is end
                self.board.position_check()
                print("nxt state", self.board.position)
                print("---------------------")
                self.restart = self.board.restart
            
            self.board.showBoard()
            
        print("Final Q Values: ")
        self.showQ_values()


# In[436]:


rows = 5
columns = 5
goal = (2,2)
start = (0,0)
obstacles = [(1,1),(0,1)]
board = Environment(COLUMNS = columns,ROWS = rows,START = start, GOAL = goal, OBSTACLES = obstacles) #COLUMNS = 3,ROWS = 3,START = (0,0) ,GOAL = (2,2),OBSTACLES = []
#board = Environment()
agent = Agent(BOARD = board, LR = 0.5, EXP_R = 0.4, GAMMA = 0.5) #


# In[437]:


agent.Q_learning(1000)


# In[ ]:




