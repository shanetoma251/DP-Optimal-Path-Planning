'''
Shane Toma | RBE595: Programming Assignment 2 | 2/11/2024
'''

import numpy as np
import matplotlib.pyplot as plt
#initilize problem parameters
#load world data
world = np.loadtxt("world.txt")
goal = (7,10)
goal_y, goal_x = goal
y_max, x_max = world.shape
#extract coordinates of world not occupied by objects into tuple for hashing
states = [tuple(coords) for coords in np.argwhere(world == 0)]

n_states = len(states)
n_actions = 8
gamma = 0.95
theta = 1e-9

#Switch between deterministic and stochastic models
deterministic = False

#dict for action states
actions = {
    "up":(-1,0),
    "down":(1,0),
    "left":(0,-1),
    "right":(0,1),
    "up_left":(-1,-1),
    "up_right":(-1,1),
    "down_left":(1,-1),
    "down_right":(1,1)
}
#Create dict with prob of moving in the same direction as directed by an action
if deterministic == True:
    #for deterministic model, this is 1 since robot always moves according to action
    action_prob = {key:[(key,1)] for key in actions.keys()}
else:
    action_prob = {"up":[("up",0.6),("up_left",0.2),("up_right",0.2)],
                   "down":[("down",0.6),("down_left",0.2),("down_right",0.2)],
                   "left":[("left",0.6),("up_left",0.2),("down_left",0.2)],
                   "right":[("right",0.6),("up_right",0.2),("down_right",0.2)],
                   "up_left":[("up_left",0.6),("up",0.2),("left",0.2)],
                   "up_right":[("up_right",0.6),("up",0.2),("right",0.2)],
                   "down_left":[("down_left",0.6),("left",0.2),("down",0.2)],
                   "down_right":[("down_right",0.6),("down",0.2),("right",0.2)],}


#init dict for policy
initial_policy = {}
init_value_function = np.zeros(world.shape)

for state in states:
    #create dict that maps a policy w p(any_move)=0.125 for each state
    initial_policy[tuple(state)] = {key:1/n_actions for key in actions.keys()}
                 
"""
Function to evaluate a policy and return value function given the initial policy, value function
states, actions, action_prob
 """
def evaluate_policy(policy,value_function,states, actions, action_prob):
    delta = 0
    for state in states: 
        v = 0
        for action in action_prob:
            #find probability of each action in a given state
            p_action = policy[state][action]

            if deterministic == True:
                #now find the next state, s' using the given action and probability of that action
                p_chosen = 1
                state_prime = tuple(map(sum, zip(state, actions[action])))
                #monitor state updates
                # print("\nstate:",state)
                # print("\naction: ",action)
                # print("\nNew state:",state_prime)

                #get reward from action based on new state

                #find if new state is in bounds
                in_bounds = False
                for test_state in states:
                    if state_prime == test_state:
                        in_bounds = True
                        break
                #calc reward for new state
                reward = find_reward(state_prime,goal,in_bounds)
                v += p_action*p_chosen*(reward+gamma*value_function[state_prime])
            
            else:
                #loop through each possible action if robot deviates from commanded action
                for stoch_action, p_chosen in action_prob[action]:
                    state_prime = tuple(map(sum, zip(state, actions[stoch_action])))

                    in_bounds = False
                    for test_state in states:
                        if state_prime == test_state:
                            in_bounds = True
                            break
                    #calc reward for new state
                    reward = find_reward(state_prime,goal,in_bounds)
                    v += p_action*p_chosen*(reward+gamma*value_function[state_prime])

        delta = max(delta,abs(v-value_function[state]))
        value_function[state] = v
        
    return value_function, delta
'''
Function to improve policy iteravely until it has stabilized. Takes inputs as current policy and value_function
Returns updated policy and value function and boolean value for stability
'''
def improve_policy(policy, value_function):
    policy_stable = True
    for state in states:
        #get the old (previously best) action at each state
        #this is the one with the highest probability
        old_action = max(policy[state], key= policy[state].get)
        action_value = {key:0 for key in actions} #init empty dict
        #loop through actions to find highest reward
        for action in actions:
            if deterministic == True:
                p_action = 1
                state_prime = tuple(map(sum, zip(state, actions[action])))
                #find if new state is in bounds
                in_bounds = False
                for test_state in states:
                    if state_prime == test_state:
                        in_bounds = True
                        break
                #calc reward for new state
                reward = find_reward(state_prime,goal,in_bounds)
                action_value[action] += p_action*(reward + gamma*value_function[state_prime])
            else:
                for stoch_action, p_action in action_prob[action]:

                    state_prime = tuple(map(sum, zip(state, actions[stoch_action])))
                    #find if new state is in bounds
                    in_bounds = False
                    for test_state in states:
                        if state_prime == test_state:
                            in_bounds = True
                            break
                    #calc reward for new state
                    reward = find_reward(state_prime,goal,in_bounds)
                    action_value[action] += p_action*(reward + gamma*value_function[state_prime])
        #find optimal action from current state
        optimal_action = max(action_value,key=action_value.get)
        #compare old action and new action
        if optimal_action != old_action:
            policy_stable=False
        #update policy with zeros for every action
        for key in policy[state]:
            policy[state][key] = 0
        #update policy with 1 for the optimal action
        policy[state][optimal_action] = 1
    
    return policy, policy_stable, value_function
'''
Function to iterate model value function. Takes current value function and returns optimal policy and corresponding value function
'''
def iterate_value(value_function):
    v = value_function
    delta = 1 #initilize delta > theta
    while True:
        delta = 0
        for state in states:
            action_value = {key:0 for key in actions}
            for action in actions:
                if deterministic == True:
                    #no need to loop through stochastic action deviations
                    p_action = 1
                    state_prime = tuple(map(sum, zip(state, actions[action])))
                    #is next state s' in bounds?
                    in_bounds = False
                    for test_state in states:
                        if state_prime == test_state:
                            in_bounds = True
                            break
                    
                    #calc reward for new state
                    reward = find_reward(state_prime,goal,in_bounds)
                    action_value[action] += p_action*(reward + gamma*v[state_prime])
                else:
                    for stoch_action, p_action in action_prob[action]:

                        state_prime = tuple(map(sum, zip(state, actions[stoch_action])))
                        #find if new state is in bounds
                        in_bounds = False
                        for test_state in states:
                            if state_prime == test_state:
                                in_bounds = True
                                break
                        #calc reward for new state
                        reward = find_reward(state_prime,goal,in_bounds)
                        action_value[action] += p_action*(reward + gamma*value_function[state_prime])

            #find optimal action from current state
            optimal_action = max(action_value.values())
            delta = max(delta,abs(optimal_action-v[state]))
            v[state] = optimal_action
        if delta < theta:
            break
    #find optimal policy with optimal value function
    policy = {}
    for state in states:
        #create dict that inits policy with zeros
        policy[state] = {key:0 for key in actions.keys()}
    
    policy, policy_stable, value_function = improve_policy(policy,v)
    return policy, value_function
'''
Generalized policy interation, iterates without waiting for delta < theta
'''
def iterate_policy_generalized():
    policy_stable = False
    while policy_stable == False:
        val_fnc, delta = evaluate_policy(initial_policy,init_value_function,states,actions,action_prob)
        policy, policy_stable, value_function = improve_policy(initial_policy ,val_fnc)
    return policy, val_fnc

'''
Calculates reward based on the current state
'''
def find_reward(state_prime,goal,in_bounds):

    if in_bounds == False:
        reward = -50
    
    else:
        if state_prime == goal:
            reward = 100
        else:
            reward = -1

    return reward

def value_plot(value_function,title):
    
    plt.title(title)
    plt.plot(goal_x,goal_y,"go")
    plt.imshow(world, cmap="binary")
    plt.imshow(value_function,cmap="gray")
    plt.show()

def policy_plot(policy,title):
    
    plt.title(title)
    plt.plot(goal_x,goal_y,"go")
    plt.imshow(world, cmap="binary")

    #plot arrows for each state
    for state in policy:
        for key in policy[state]:
            
            y,x = state
            dy, dx = 0, 0
            if policy[state][key] == 1:
                
                if key == "up":
                    dy = -0.5
                elif key == "down":
                    dy = 0.5
                elif key == "left":
                    dx = -0.5
                elif key == "right":
                    dx = 0.5
                elif key == "up_left":
                    dy = -0.25
                    dx = -0.25
                elif key == "up_right":
                    dy = -0.25
                    dx = 0.25
                elif key == "down_left":
                    dy = 0.25
                    dx = -0.25
                elif key == "down_right":
                    dy = 0.25
                    dx = 0.25
            
                plt.arrow(x,y,dx,dy,width=0.05)
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# now we can iterate our policy
policy = initial_policy
val_fnc = init_value_function
policy_stable = False
#iterate until policy improvement is complete
count = 0
while policy_stable == False:
    delta = 0
    while delta < theta:                    
        val_fnc, delta = evaluate_policy(policy,val_fnc,states,actions,action_prob)
    
    policy, policy_stable, value_function = improve_policy(initial_policy ,val_fnc)
    count += 1


#plotting
#First print deterministic policy and value plots for policy iteration
deterministic = True
policy_plot(policy, "Control Policy for Policy Iteration with Deterministic Model")
value_plot(value_function, "Value Function for Policy Iteration an MDP Deterministic Model")
#now print for generalized iteration
policy, value_function = iterate_policy_generalized()
policy_plot(policy, "Control Policy for Generalized Policy Iteration with Deterministic Model")
value_plot(value_function, "Value Function for Generalized Policy Iteration Deterministic Model")
#now plot value iteration
policy, value_function = iterate_value(init_value_function)
policy_plot(policy, "Control Policy for Value Iteration with Deterministic Model")
value_plot(value_function, "Value Function for Value Iteration Deterministic Model")

#now repeat with stochastic model
policy = initial_policy
val_fnc = init_value_function
policy_stable = False
#iterate until policy improvement is complete
count = 0
while policy_stable == False:
    delta = 0
    while delta < theta:                    
        val_fnc, delta = evaluate_policy(policy,val_fnc,states,actions,action_prob)
    
    policy, policy_stable, value_function = improve_policy(initial_policy ,val_fnc)

deterministic = False
policy_plot(policy, "Control Policy for Policy Iteration with Stochastic Model")
value_plot(value_function, "Value Function for Policy Iteration with Stochastic Model")
#now print for generalized iteration
policy, value_function = iterate_policy_generalized()
policy_plot(policy, "Control Policy for Generalized Policy Iteration with Stochastic Model")
value_plot(value_function, "Value Function for Generalized Policy Iteration with Stochastic Model")
#now plot value iteration
policy, value_function = iterate_value(init_value_function)
policy_plot(policy, "Control Policy for Value Iteration with Stochastic Model")
value_plot(value_function, "Value Function for Value Iteration with Stochastic Model")

