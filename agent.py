############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
import collections
import matplotlib.pyplot as plt


class Agent:
    MIN_NUMBER_OF_BATCHES = 50

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length (you will need to increase this)
        self.episode_length = 20
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # The deep q network will be trained on agent's experience
        self.dqn = DQN()
        # The agent will use epsilon-greedy policy as a trade-off between exploration and exploitation
        self.epsilon = 0.2
        # The agent stores its past transitions into a replay buffer
        self.replay_buffer = ReplayBuffer(max_capacity=1000000)

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Compute the Q-values for this state
        q_values_tensor = self.dqn.q_network.forward(torch.tensor(np.array([state])).float())
        # Determine the action that leads to the highest Q-value in this state
        discrete_action = np.random.choice(np.arange(4), 1, p=np.array([(1 - self.epsilon) if i == q_values_tensor.max(1)[1].item() else (self.epsilon / 3) for i in range(4)]))[0]
        action = self._discrete_action_to_continuous(discrete_action=discrete_action)
        print(action)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = discrete_action
        return action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        print(next_state)
        # Convert the distance to a reward
        reward = 1 - distance_to_goal
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        # Add transition to the agent's replay buffer
        self.replay_buffer.add_sample(transition)
        # If at least one batch can be extracted from the buffer, train the network on a random batch
        if len(self.replay_buffer) >= Agent.MIN_NUMBER_OF_BATCHES:
            mini_batch = self.replay_buffer.random_batch(N=Agent.MIN_NUMBER_OF_BATCHES)
            loss = self.dqn.train_q_network(mini_batch)

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        q_values_tensor = self.dqn.q_network.forward(torch.tensor(np.array([state])).float())
        action = q_values_tensor.max(1)[1].item()  # index of the max q value represents the greedy action
        return action

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:  # Move up
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        elif discrete_action == 1:  # Move right
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        elif discrete_action == 2:  # Move down
            continuous_action = np.array([0, -0.02], dtype=np.float32)
        else:  # Move left
            continuous_action = np.array([-0.02, 0], dtype=np.float32)
        return continuous_action


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the
    # dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example,
    # a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it
    # is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self, discount=0.9, learning_rate=0.001):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Create a target Q-network which will be used in the Bellman equation
        self.target_q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each
        # gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        # discount factor
        self.discount = discount
        #
        self.set_actions = np.arange(0, 360, 30)  # angles of possible actions

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a
    # transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, batch):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(batch)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network
        # parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    def _calculate_loss(self, transitions):
        print(transitions)
        inputs = np.array([transitions[i, 0] for i in range(transitions.shape[0])])
        rewards = np.array([np.array([(transitions[i, 2])]) for i in range(transitions.shape[0])])
        next_state_inputs = np.array([transitions[i, 3] for i in range(transitions.shape[0])])

        batch_inputs_tensor = torch.tensor(inputs).float()
        batch_rewards_tensor = torch.tensor(rewards).float()
        batch_actions_tensor = torch.tensor(
            np.array([transitions[i, 1] for i in range(transitions.shape[0])])).unsqueeze(1)
        next_state_inputs_tensor = torch.tensor(next_state_inputs).float()
        next_state_target_q_values = self.q_network.forward(next_state_inputs_tensor)
        network_q_values = self.q_network.forward(batch_inputs_tensor)
        # print('label ', batch_rewards_tensor)
        print('q values', network_q_values.gather(1, batch_actions_tensor).shape)
        print('reward', batch_rewards_tensor.shape)
        print('actions', batch_actions_tensor.shape)
        print('target q values max', next_state_target_q_values.max(1)[0].unsqueeze(
                                      1).shape)
        loss = torch.nn.MSELoss()(network_q_values.gather(1, batch_actions_tensor),
                                  batch_rewards_tensor + self.discount * next_state_target_q_values.max(1)[0].unsqueeze(
                                      1))
        return loss

    # # Function to calculate the loss for a particular transition.
    # def _calculate_loss(self, batch):
    #     inputs = np.array([[batch[i, 0], batch[i, 1]] for i in range(batch.shape[0])])
    #     batch_inputs_tensor = torch.tensor(inputs).float()
    #     network_q_values = self.q_network.forward(batch_inputs_tensor)
    #
    #     rewards = np.array([np.array([(batch[i, 2])]) for i in range(batch.shape[0])])
    #     batch_rewards_tensor = torch.tensor(rewards).float()
    #
    #     next_state_inputs = np.array([batch[i, 3] for i in range(batch.shape[0])])
    #     next_state_inputs_tensor = torch.tensor(next_state_inputs).float()
    #     next_state_target_q_values = self.target_q_network.forward(next_state_inputs_tensor)
    #
    #     loss = torch.nn.MSELoss()(network_q_values, batch_rewards_tensor +
    #                               self.discount * next_state_target_q_values.max(1)[0].unsqueeze(1))
    #     return loss

    # Function which copies the Q network weights into the target Q network
    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    # TODO: Not sure if it is useful in our case
    def cross_entropy(self, state):
        number_of_sampled_actions = 6
        number_of_highest_q_values_retained = 4
        distribution = lambda low, high, size: np.random.uniform(low=low, high=high, size=size)
        while True:
            sampled_actions = distribution(self.set_actions[0], self.set_actions[-1], number_of_sampled_actions)
            q_values = [self.q_network.forward(np.array([[state, action] for action in sampled_actions]))]


class ReplayBuffer:
    def __init__(self, max_capacity):
        self.replay_buffer = collections.deque(maxlen=max_capacity)

    def add_sample(self, sample):
        self.replay_buffer.append(sample)

    def random_batch(self, N):
        random_indices = np.random.choice(range(len(self.replay_buffer)), size=N)
        random_batch = np.array([self.replay_buffer[i] for i in random_indices])
        return random_batch

    def __len__(self):
        return len(self.replay_buffer)
