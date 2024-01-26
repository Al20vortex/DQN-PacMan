import gymnasium as gym
from collections import deque
from network import QNetwork
import random
import torch
from torchvision import transforms
import copy
import os

NUM_EPISODES = 1000
EPSILON = 0.05
FRAME_STACK_SIZE = 4
BATCH_SIZE = 32

device = torch.device('mps')

def choose_action(state, env, q_network):
    if random.random() < EPSILON:
        return env.action_space.sample()
    else:
        output = q_network(state)
        return torch.argmax(output).item()

def replay(replay_buffer: deque, q_network, target_q_network, optimizer, gamma=0.99):
    if len(replay_buffer) < BATCH_SIZE:
        return
    
    experiences = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, terminals = zip(*experiences)

    states = torch.cat(states).to(device)
    actions = torch.tensor(actions).to(device)
    rewards = torch.tensor(rewards).to(device)
    next_states = torch.cat(next_states).to(device)
    terminals = torch.tensor(terminals).to(device)

    # Get Q values for current states
    current_q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Get Q values for next states
    next_q_values = target_q_network(next_states).max(1)[0]
    next_q_values[terminals] = 0.  # Terminal states should be 0

    # Compute the target Q values
    target_q_values = rewards + gamma * next_q_values

    # Compute loss and optimize the model
    loss = torch.nn.MSELoss()(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

env = gym.make("MsPacman-v4", render_mode=None)

# Initialize replay buffer
replay_buffer = deque(maxlen=30000)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((84, 84)), # TODO should actually be resized and then cropped to 84x84
    transforms.ToTensor() 
])

# Initialize action-value function Q with random weights
q_network = QNetwork(env.action_space.n).to(device)

# Initialize target action-value function Q' with weights of Q
target_q_network = QNetwork(env.action_space.n).to(device)
target_q_network.load_state_dict(q_network.state_dict())

saved_model_path = 'saved_best_model.pt'
if os.path.exists(saved_model_path):
    q_network.load_state_dict(torch.load(saved_model_path, map_location=device))
    target_q_network.load_state_dict(torch.load(saved_model_path, map_location=device))

# Optimizer for Q-Network
optimizer = torch.optim.Adam(q_network.parameters(), lr=0.001)
for episode in range(NUM_EPISODES):
    observation, info = env.reset()
    observation_tensor = transform(observation).unsqueeze(0).to(device)
    frame_stack = deque(maxlen=FRAME_STACK_SIZE)
    for _ in range(FRAME_STACK_SIZE):
        frame_stack.append(observation_tensor)
    # initialize frame stack
    terminal = False
    frame = 0
    score = 0
    best_score = 0
    while not terminal:
        frame += 1
        state = torch.cat(list(frame_stack), dim=1).to(device)  # Stack the frames
        action = choose_action(state, env, q_network)
        new_observation, reward, terminated, truncated, info = env.step(action)
        new_observation_tensor = transform(new_observation).unsqueeze(0).to(device)
        frame_stack.append(new_observation_tensor)
        next_state = torch.cat(list(frame_stack), dim=1)  # Stack the frames
        terminal = terminated or truncated
        score+=reward

        # Store experience in replay buffer
        replay_buffer.append((state, action, reward, next_state, terminal))
        
        replay(replay_buffer, q_network, target_q_network, optimizer)

        # Periodically update the target network for stability
        if frame % 3 == 0:
            target_network = copy.deepcopy(q_network).to(device)
            torch.save(target_network.state_dict(), 'saved_model.pt')
        if score > best_score:
            best_score = score
            torch.save(q_network.state_dict(), 'saved_best_model.pt')
    
    print(f"Episode score: {score}")
