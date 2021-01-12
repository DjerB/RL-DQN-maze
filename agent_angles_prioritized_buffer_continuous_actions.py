import numpy as np
import torch
import collections
import matplotlib.pyplot as plt
import math
import random
import itertools


class Agent:
    MAX_ACTION_NORM = 0.02
    # Function to initialise the agent
    def __init__(self):
        # Set the episode length (you will need to increase this)
        self.set_angles_actions = None
        self.episode_length = 400
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # The deep q network will be trained on agent's experience
        self.dqn = DQN()
        # The agent will use epsilon-greedy policy as a trade-off between exploration and exploitation
        self.epsilon = 0.9
        self.decay = 0.0000048
        # The agent can move in a set of different directions
        self.range_angles_actions = [0, 360]  # angles of possible actions

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        self.epsilon = self.decay - self.epsilon if self.decay - self.epsilon > 0.2 else 0.2
        return self.num_steps_taken % self.episode_length == 0

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        #inputs = np.array([np.append(state, self._discrete_action_to_continuous(i)) for i in range(0, len(self.set_angles_actions))])
        # Compute the Q-values for this state
        #q_values_tensor = self.dqn.q_network.forward(torch.tensor(inputs).float())
        # Determine the action that leads to the highest Q-value in this state
        proba_epsilon_greedy = random.random()
        if proba_epsilon_greedy < self.epsilon:
            angle_action = random.uniform(self.range_angles_actions[0], self.range_angles_actions[1])
        else:
            self.dqn.cross_entropy(state)
            greedy_angle_action = self.dqn.cross_entropy_distribution.mean
            angle_action = greedy_angle_action
        action = self._angle_action_to_continuous(angle_action)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        # reward = 1 - distance_to_goal
        #reward = 1 - distance_to_goal
        reward = 100 if distance_to_goal == 0 else - distance_to_goal ** 2 if distance_to_goal > 0.4 else 1 - distance_to_goal
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        # Add transition to the agent's replay buffer
        self.dqn.replay_buffer.add_sample(transition)
        # If at least one batch can be extracted from the buffer, train the network on a random batch
        if len(self.dqn.replay_buffer) >= ReplayBuffer.MIN_NUMBER_OF_BATCHES:
            mini_batch = self.dqn.replay_buffer.get_batch(N=ReplayBuffer.MIN_NUMBER_OF_BATCHES)
            loss = self.dqn.train_q_network(mini_batch)
        if self.num_steps_taken % DQN.NUMBER_OF_STEPS_BEFORE_TARGET_UPDATE == 0:
            self.dqn.update_target_q_network()

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        # q_values_tensor = self.dqn.q_network.forward(torch.tensor(np.array([state])).float())
        # action = q_values_tensor.max(1)[1].item()  # index of the max q value represents the greedy action
        # inputs = np.array([np.append(state, self._discrete_action_to_continuous(i)) for i in range(0, len(self.set_angles_actions))])
        # # Compute the Q-values for this state
        # q_values_tensor = self.dqn.q_network.forward(torch.tensor(inputs).float())
        # action = self._angle_action_to_continuous(discrete_action=q_values_tensor.max(0)[1].item())
        self.dqn.cross_entropy(state)
        greedy_angle_action = self.dqn.cross_entropy_distribution.mean
        action = self._angle_action_to_continuous(greedy_angle_action)
        return action

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _angle_action_to_continuous(self, angle):
        angle = math.radians(angle)
        continous_action = np.array([math.cos(angle) * Agent.MAX_ACTION_NORM, math.sin(angle) * Agent.MAX_ACTION_NORM], dtype=np.float32)
        return continous_action


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
        self.layer_3 = torch.nn.Linear(in_features=100, out_features=64)
        self.output_layer = torch.nn.Linear(in_features=64, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example,
    # a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it
    # is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:
    NUMBER_OF_STEPS_BEFORE_TARGET_UPDATE = 1000

    # The class initialisation function.
    def __init__(self, discount=0.7, learning_rate=0.01):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=4, output_dimension=1)
        # Create a target Q-network which will be used in the Bellman equation
        self.target_q_network = Network(input_dimension=4, output_dimension=1)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each
        # gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        # discount factor
        self.discount = discount
        #
        self.range_angles_actions = [0, 360]  # angles of possible actions
        # The q network stores the agent past transitions into a replay buffer
        self.replay_buffer = ReplayBuffer(max_capacity=1000000)

        self.cross_entropy_distribution = GaussianDistribution((self.range_angles_actions[0] + self.range_angles_actions[1])/2, len(self.range_angles_actions))

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

    def _calculate_loss(self, batch):
        # Inputs of the network are the state and the continuous action
        inputs = np.array([np.append(batch[i, 0], batch[i, 1]) for i in range(batch.shape[0])])
        rewards = np.array([np.array([(batch[i, 2])]) for i in range(batch.shape[0])])

        next_state_max_values = []
        for next_state_index in range(batch.shape[0]):
            self.cross_entropy(batch[next_state_index, 3], use_target=True)
            next_state_max_values.append(self.cross_entropy_distribution.mean)
        next_state_max_values = torch.tensor(next_state_max_values).unsqueeze(1)

        batch_inputs_tensor = torch.tensor(inputs).float()
        batch_rewards_tensor = torch.tensor(rewards).float()

        network_q_values = self.q_network.forward(batch_inputs_tensor)
        # print('label ', batch_rewards_tensor)
        loss = torch.nn.MSELoss()(network_q_values, batch_rewards_tensor + self.discount * next_state_max_values)

        next_state_max_values_array = np.array([next_state_max_values[i].item() for i in range(next_state_max_values.shape[0])])
        batch_rewards_array = np.array([batch_rewards_tensor[i].item() for i in range(batch_rewards_tensor.shape[0])])
        deltas = batch_rewards_array + self.discount * next_state_max_values_array
        self.replay_buffer.update_transitions_weights(deltas)
        return loss

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, angle_action):
        angle = math.radians(angle_action)
        continuous_action = np.array(
            [math.cos(angle) * Agent.MAX_ACTION_NORM, math.sin(angle) * Agent.MAX_ACTION_NORM], dtype=np.float32)
        return continuous_action

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
    def cross_entropy(self, state, use_target=False):
        number_of_sampled_actions = 10  # M in the lectures
        number_of_highest_q_values_retained = 4  # N in the lecture
        distribution = lambda low, high, size: np.random.uniform(low=low, high=high, size=size)
        last_mean = float('inf')
        while True:
            sampled_actions = distribution(self.range_angles_actions[0], self.range_angles_actions[1], number_of_sampled_actions)
            states = [state] * len(sampled_actions)
            inputs = np.array(
                [[states[i][0], states[i][1], self._discrete_action_to_continuous(angle_action)[0], self._discrete_action_to_continuous(angle_action)[1]] for i, angle_action in
                 enumerate(sampled_actions)])
            inputs_tensor = torch.tensor(inputs)
            if use_target:
                q_values = self.target_q_network.forward(inputs_tensor).detach()
                q_values = [value.item() for value in q_values]
            else:
                q_values = self.q_network.forward(inputs_tensor).detach()
                q_values = [value.item() for value in q_values]
            top_values = sorted(list(zip(sampled_actions, q_values)), key=lambda x: x[1], reverse=True)[:number_of_highest_q_values_retained]
            mean = sum([t[0] for t in top_values]) / len(top_values)
            std = math.sqrt(sum((t[0] - mean) ** 2 for t in top_values) / len(top_values))
            distribution = GaussianDistribution(mean, std)
            if abs(last_mean - mean) < 3:
                break
            last_mean = mean
        self.cross_entropy_distribution = distribution


class GaussianDistribution:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self, low=None, high=None, size=1):
        return np.random.normal(self.mean, self.std, size)

    def __call__(self, *args, **kwargs):
        return self.sample(args)

class ReplayBuffer:
    MIN_NUMBER_OF_BATCHES = 300
    def __init__(self, max_capacity):
        self.replay_buffer = collections.deque(maxlen=max_capacity)
        self.current_batch = None
        self.bias_weight = 0.01
        self.alpha = 3
        self.transitions_weights = []
        self.sampling_probabilities = []
        self.update_count = 0

    def add_sample(self, sample):
        self.replay_buffer.append(sample)
        self.transitions_weights.append(0)
        if len(self) == ReplayBuffer.MIN_NUMBER_OF_BATCHES:
            self.sampling_probabilities = [1 / len(self)] * len(self)
        if len(self) > ReplayBuffer.MIN_NUMBER_OF_BATCHES:
            self.sampling_probabilities.append(max(self.sampling_probabilities))
        self.update_count += 1

    def get_batch(self, N):
        batch_indices = sorted(range(len(self.sampling_probabilities)), key=lambda i: self.sampling_probabilities[i], reverse=True)[:N]
        # random_indices = np.random.choice(range(len(self.replay_buffer)), size=N)
        batch = np.array([self.replay_buffer[i] for i in batch_indices])
        self.batch_indices = batch_indices
        return batch

    def __len__(self):
        return len(self.replay_buffer)

    def update_transitions_weights(self, deltas):
        batch_weights = np.abs(deltas) + self.bias_weight
        for k, i in enumerate(self.batch_indices):
            self.transitions_weights[i] = batch_weights[k].item()
        self.update_sampling_probabilities()

    def update_sampling_probabilities(self):
        sum_of_weights = sum([weight ** self.alpha for weight in self.transitions_weights])
        for i in range(len(self.sampling_probabilities)):
            self.sampling_probabilities[i] = self.transitions_weights[i] ** self.alpha / sum_of_weights

        if self.update_count > 2 * ReplayBuffer.MIN_NUMBER_OF_BATCHES:
            cutoff = int(len(self)/3)
            self.replay_buffer = collections.deque(itertools.islice(self.replay_buffer, cutoff, len(self)))
            self.transitions_weights = self.transitions_weights[cutoff:]
            self.sampling_probabilities = self.sampling_probabilities[cutoff:]
            self.update_count = 0
