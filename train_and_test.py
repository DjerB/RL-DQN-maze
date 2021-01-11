import time
import numpy as np
import cv2
from random_environment import Environment
from agent_final_2 import Agent

def plot_policy(agent, environment):


    lines = []
    state = environment.init_state
    for step_num in range(100):
        action = agent.get_greedy_action(state)
        next_state, distance_to_goal = environment.step(state, action)
        lines.append((state, next_state))
        state = next_state

    total_path_length = len(lines)
    cumul = 0
    for line in lines:
        start = (int(line[0][0] * environment.magnification), int((1 - line[0][1]) * environment.magnification))
        end = (int(line[1][0] * environment.magnification), int((1 - line[1][1]) * environment.magnification))
        gradient = (cumul / total_path_length)
        colour = (0, gradient * 255, int((1 - gradient) * 255))
        cv2.line(environment.image, start, end, color=colour, thickness=2)
        cumul += 1

    cv2.imshow("Environment", environment.image)
    cv2.waitKey(1)

# Main entry point
if __name__ == "__main__":

    # This determines whether the environment will be displayed on each each step.
    # When we train your code for the 10 minute period, we will not display the environment.
    display_on = True

    # Create a random seed, which will define the environment
    random_seed = int(time.time())
    np.random.seed(random_seed)

    # Create a random environment
    environment = Environment(magnification=500)
    policy_environment = Environment(magnification=500)

    # Create an agent
    agent = Agent()

    # Get the initial state
    state = environment.init_state

    # Determine the time at which training will stop, i.e. in 10 minutes (600 seconds) time
    start_time = time.time()
    end_time = start_time + 600

    # Train the agent, until the time is up
    while time.time() < end_time:
        # If the action is to start a new episode, then reset the state
        if agent.has_finished_episode():
            state = environment.init_state
        # Get the state and action from the agent
        action = agent.get_next_action(state)
        # Get the next state and the distance to the goal
        next_state, distance_to_goal = environment.step(state, action)
        # Return this to the agent
        agent.set_next_state_and_distance(next_state, distance_to_goal)
        #plot_policy(agent, environment)

        state = next_state
        if display_on:
            environment.show(state)

    # Test the agent for 100 steps, using its greedy policy
    state = environment.init_state
    has_reached_goal = False
    for step_num in range(100):
        action = agent.get_greedy_action(state)
        next_state, distance_to_goal = environment.step(state, action)
        # The agent must achieve a maximum distance of 0.03 for use to consider it "reaching the goal"
        if distance_to_goal < 0.03:
            has_reached_goal = True
            break
        state = next_state

    # Print out the result
    if has_reached_goal:
        print('Reached goal in ' + str(step_num) + ' steps.')
    else:
        print('Did not reach goal. Final distance = ' + str(distance_to_goal))
