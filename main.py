import gymnasium as gym
from collections import deque
from network import QNetwork
import random
import torch
from torchvision import transforms
import copy
import os
from utils import *
import wandb
from gymnasium.wrappers import RecordVideo

NUM_EPISODES = 1000000
EPSILON = 1.0
FRAME_STACK_SIZE = 4
BATCH_SIZE = 64
# For recording
RECORD_INTERVAL = 1000  # How often to record

device = get_device()
def choose_action(state, env, q_network):
    if random.random() < EPSILON:
        return env.action_space.sample()
    else:
        output = q_network(state)
        return torch.argmax(output).item()

def replay(replay_buffer: deque, q_network, target_q_network, optimizer, gamma=0.999):
    if len(replay_buffer) < replay_buffer.maxlen//2:
        return
    experiences = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, terminals = zip(*experiences)

    states = torch.cat(states).to(device)
    actions = torch.tensor(actions).to(device)
    rewards = torch.tensor(rewards).to(device)
    next_states = torch.cat(next_states).to(device)
    terminals = torch.tensor(terminals, dtype=torch.float32).to(device)
    # Get Q values for current states
    current_q_values = q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    # Get Q values for next states
    with torch.no_grad():
        next_q_values = torch.amax(target_q_network(next_states), dim=1)
        next_q_values = next_q_values @ (1-terminals)  # zero out the terminal states

    # Compute the target Q values
    target_q_values = rewards + gamma * next_q_values

    # Compute loss and optimize the model
    loss = torch.nn.HuberLoss()(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# env = gym.make("MsPacman-v4", render_mode=None)
env = gym.make("Breakout-v4", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, 'videos/', episode_trigger=lambda x: x % RECORD_INTERVAL == 0, video_length=0)

# Initialize replay buffer
replay_buffer = deque(maxlen=100000)

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
optimizer = torch.optim.Adam(q_network.parameters(), lr=5e-4)

wandb.login()
wandb.init(project="DQN-Breakout", mode="online")

# Initialize steps count
steps = 0

# Main Training Loop
for episode in range(NUM_EPISODES):
    observation, info = env.reset()
    observation_tensor = transform(observation).unsqueeze(0).to(device)

    # initialize frame stack
    frame_stack = deque(maxlen=FRAME_STACK_SIZE)
    for _ in range(FRAME_STACK_SIZE):
        frame_stack.append(observation_tensor)

    # Initialize counters
    terminal = False
    frame = 0
    score = 0
    best_score = 0
    
    while not terminal:
        # Linear epsilon annealing over 1 million steps (rougly)
        EPSILON = 1.0 - steps/1000000
        if EPSILON < 0.1:
            EPSILON = 0.1
        if episode % 100 == 0:
            EPSILON = 0.0

        steps+=1 
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
        if steps % 10000 == 0:
            target_network = copy.deepcopy(q_network).to(device)
            torch.save(target_network.state_dict(), 'saved_model.pt')
    if score > best_score:
        best_score = score
        torch.save(q_network.state_dict(), 'saved_best_model.pt')

    print(f"Episode score: {score}, Epsilon: {EPSILON}")
    if episode % RECORD_INTERVAL == 0:
        wandb.log({
        "Episode Score": score,
        "Test Score": score,
        "Epsilon": EPSILON,
        "Num Steps": steps
        })
    else:
        wandb.log({
        "Episode Score": score,
        "Epsilon": EPSILON,
        "Num Steps": steps
        })

