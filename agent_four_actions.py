import numpy as np
import torch
import collections
import time
import itertools


class Agent:
    # Function to initialise the agent
    def __init__(self):
        # Set the episode length (you will need to increase this)
        self.episode_length = 1400
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
        # The epsilon value will be decreased over time to focus on the greedy policy and strengthen it
        self.decay = 0.0003
        # Start time for the agent
        self.start = time.time()

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        '''
        Determines the completion of the current episode. Also implements a specific stragegy on the epsilon value.
        Epsilon follows the following pattern:
        - Freezes for the first 2 minutes for exploration purposes
        - Starts decreasing to 0.75 (4 actions with same probability) for 2 minutes
        - Keeps decreasing to 0.6 for 4 minutes (greedy policy is followed for the first steps and exploration
        keeps going on
        - Keeps decreasing to its minimal value 0.2: the greedy policy should be well defined at this point and
        the idea is to confirm this policy by evaluating its surrounding states

        Similarly, the episode length is decreased over time as exploration is little by little replaced with
        exploitation but not completely
        :return:
        '''
        if 2000 < self.num_steps_taken < 4000:
            self.epsilon = self.epsilon - self.decay if self.epsilon - self.decay > 0.75 else 0.75
        elif 4000 <= self.num_steps_taken <= 8000:
            self.epsilon = self.epsilon - self.decay if self.epsilon - self.decay > 0.6 else 0.6
        elif 8000 < self.num_steps_taken:
            self.epsilon = self.epsilon - self.decay if self.epsilon - self.decay > 0.2 else 0.2
        self.episode_length = int(self.episode_length * 0.99) if int(self.episode_length * 0.99) >= 280 else self.episode_length
        return self.num_steps_taken % self.episode_length == 0

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Compute the Q-values for this state
        q_values_tensor = self.dqn.q_network.forward(torch.tensor(np.array([state]).astype(np.float32)))
        # Determine the action that leads to the highest Q-value in this state
        greedy_action = q_values_tensor.max(1)[1].item()
        # For the first 3 minutes, just explores randomly the environment without taking the greedy action into account
        if self.num_steps_taken > 3000:
            discrete_action = np.random.choice(np.arange(4), 1, p=np.array(
                [(1 - self.epsilon) if i == greedy_action else (self.epsilon / 3) for i in
                 range(4)]))[0]
        else:
            discrete_action = np.random.choice(np.arange(4), 1, p=np.array(
                [1/4 for i in
                 range(4)]))[0]
        print(f'State: {state} - Action: {discrete_action} - Greedy: {greedy_action} - Epsilon: {self.epsilon}')
        action = self._discrete_action_to_continuous(discrete_action=discrete_action)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        print(f'Step {self.num_steps_taken}')
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = discrete_action
        return action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward. A penalty is given if the agent stands still, ie hits a wall
        if next_state[0] == self.state[0] and next_state[1] == self.state[1]:
            reward = 1 - distance_to_goal - 0.08
        else:
            reward = 1 - distance_to_goal
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        # Add transition to the agent's replay buffer
        self.dqn.replay_buffer.add_sample(transition)
        # If at least one batch can be extracted from the buffer, train the network on a random batch
        if len(self.dqn.replay_buffer) >= ReplayBuffer.MIN_NUMBER_OF_BATCHES:
            mini_batch = self.dqn.replay_buffer.get_batch()
            loss = self.dqn.train_q_network(mini_batch)
        if self.num_steps_taken % DQN.NUMBER_OF_STEPS_BEFORE_TARGET_UPDATE == 0:
            self.dqn.update_target_q_network()

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # We get the greedy action by computing the q values for this state and keeping the best one
        q_values_tensor = self.dqn.q_network.forward(torch.tensor(np.array([state])).float())
        action = q_values_tensor.max(1)[1].item()  # index of the max q value represents the greedy action
        # Compute the Q-values for this state
        action = self._discrete_action_to_continuous(discrete_action=action)
        return action

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        '''
        Returns a vector for a discrete action. Here, the amplitude depends on the direction: 0.02 for Up, Right and
        Down. Left is assigned a smaller amplitude because it is not likely to lead to the goal state but it mmight
        still be useful to get away from a deadlock during training
        :param discrete_action:
        :return:
        '''
        if discrete_action == 0:  # Move up
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        elif discrete_action == 1:  # Move right
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        elif discrete_action == 2:  # Move down
            continuous_action = np.array([0, -0.02], dtype=np.float32)
        else:  # Move left
            continuous_action = np.array([-0.005, 0], dtype=np.float32)
        return continuous_action


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the
    # dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=256)
        self.layer_2 = torch.nn.Linear(in_features=256, out_features=256)
        self.layer_3 = torch.nn.Linear(in_features=256, out_features=256)
        self.layer_4 = torch.nn.Linear(in_features=256, out_features=256)
        self.layer_5 = torch.nn.Linear(in_features=256, out_features=128)
        self.output_layer = torch.nn.Linear(in_features=128, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example,
    # a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it
    # is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        layer_4_output = torch.nn.functional.relu(self.layer_4(layer_3_output))
        layer_5_output = torch.nn.functional.relu(self.layer_5(layer_4_output))
        output = self.output_layer(layer_5_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:
    NUMBER_OF_STEPS_BEFORE_TARGET_UPDATE = 300
    # The class initialisation function.
    def __init__(self, discount=0.9, learning_rate=0.01):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Create a target Q-network which will be used in the Bellman equation
        self.target_q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each
        # gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        # discount factor
        self.discount = discount
        # The q network stores the agent past transitions into a replay buffer
        self.replay_buffer = ReplayBuffer(max_capacity=1000000)

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
        inputs = np.array([transitions[i, 0] for i in range(transitions.shape[0])])
        rewards = np.array([transitions[i, 2] for i in range(transitions.shape[0])])
        next_state_inputs = np.array([transitions[i, 3] for i in range(transitions.shape[0])])

        batch_inputs_tensor = torch.tensor(inputs).float()
        batch_rewards_tensor = torch.tensor(rewards).float().unsqueeze(1)
        batch_actions_tensor = torch.tensor(
            np.array([transitions[i, 1] for i in range(transitions.shape[0])])).unsqueeze(1)
        next_state_inputs_tensor = torch.tensor(next_state_inputs).float()

        # Forward inputs through networks
        next_state_target_q_values = self.target_q_network.forward(next_state_inputs_tensor).detach()
        #double_q_argmax_values = torch.tensor(np.array([t.max(0)[1].item() for t in next_state_target_q_values])).unsqueeze(1)
        #double_q_values = self.q_network.forward(next_state_inputs_tensor).detach().gather(1, double_q_argmax_values)

        # next_state_target_q_values = self.target_q_network.forward(next_state_inputs_tensor).detach().max(0)[1] DOUBLE Q
        network_q_values = self.q_network.forward(batch_inputs_tensor)

        loss = torch.nn.MSELoss()(network_q_values.gather(1, batch_actions_tensor),
                                   batch_rewards_tensor + self.discount * next_state_target_q_values.max(1)[0].unsqueeze(
                                       1))
        #loss = torch.nn.MSELoss()(network_q_values.gather(1, batch_actions_tensor),
        #                          batch_rewards_tensor + self.discount * double_q_values)

        deltas = (batch_rewards_tensor + self.discount * next_state_target_q_values.max(1)[0].unsqueeze(
            1) - network_q_values.gather(1, batch_actions_tensor)).detach().numpy()

        self.replay_buffer.update_transitions_weights(deltas)

        return loss

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:  # Move up
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        elif discrete_action == 1:  # Move right
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        elif discrete_action == 2:  # Move down
            continuous_action = np.array([0, -0.02], dtype=np.float32)
        else:  # Move left
            continuous_action = np.array([-0.005, 0], dtype=np.float32)
        return continuous_action

    # Function which copies the Q network weights into the target Q network
    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

class ReplayBuffer:
    MIN_NUMBER_OF_BATCHES = 300

    def __init__(self, max_capacity):
        self.replay_buffer = collections.deque(maxlen=max_capacity)
        self.current_batch = None
        self.bias_weight = 0.01
        self.alpha = 1
        self.transitions_weights = []
        self.sampling_probabilities = []
        self.update_count = 0

    def add_sample(self, sample):
        self.replay_buffer.append(sample)
        if len(self) > 3000:
            for i in range(int((len(self)/2))):
                self.replay_buffer.popleft()
                #self.replay_buffer = collections.deque(itertools.islice(self.replay_buffer, int(len(self)/4), len(self)))
        self.transitions_weights.append(0)
        if len(self) == ReplayBuffer.MIN_NUMBER_OF_BATCHES:
            self.sampling_probabilities = [1 / len(self)] * len(self)
        if len(self) > ReplayBuffer.MIN_NUMBER_OF_BATCHES:
            self.sampling_probabilities.append(max(self.sampling_probabilities))
        self.update_count += 1

    def get_batch(self):
        batch_indices = sorted(range(len(self.sampling_probabilities)), key=lambda i: self.sampling_probabilities[i], reverse=True)[:ReplayBuffer.MIN_NUMBER_OF_BATCHES]
        # batch_indices = np.random.choice(range(len(self.replay_buffer)), size=N)
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
            cutoff = int(len(self)/2)
            self.replay_buffer = collections.deque(itertools.islice(self.replay_buffer, cutoff, len(self)))
            self.transitions_weights = self.transitions_weights[cutoff:]
            self.sampling_probabilities = self.sampling_probabilities[cutoff:]
            self.update_count = 0
