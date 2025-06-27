# Space Shooter RL Agent (DQN)

## 🧠 1. Problem Overview
This project aims to apply reinforcement learning (RL) to train an intelligent agent capable of making autonomous decisions in a 2D space shooter game to achieve the highest possible score.  
The player must dodge falling obstacles of varying speeds, destroy them, and collect power-up items to extend survival time and enhance attack power.  
The implementation is based on the Deep Q-Network (DQN) framework, with a carefully designed state representation and reward mechanism to help the agent learn effective strategies for survival and combat.

---

## 🛠️ 2. Environment Setup
This project is developed using Python and incorporates tools such as Pygame, PyTorch, NumPy, and OpenCV:

- **Pygame**: Handles game visuals and logic control  
- **PyTorch**: Constructs and trains the DQN model  
- **NumPy**: Performs numerical operations and distance calculations  
- **OpenCV**: Processes visual frames, such as converting game screens to grayscale  

A 2D space shooter game environment was built alongside the reinforcement learning training pipeline.  
The project follows a modular structure, allowing for quick replacement of models, environments, and training strategies.  
During training, both reward and loss data are recorded in `.npy` files for subsequent analysis and visualization.

---

## 📂 3. Project Structure Explanation

### `game_core/`
Contains core components required for the game to run: game structure, rules, character behaviors, and collision logic.  
This folder is treated as a black box during training—used only for reading information such as player status and enemy positions.

### `envs/space_shooter_env.py`
Wraps the game into a reinforcement learning environment (similar to OpenAI Gym format). Includes:

- State initialization and frame stacking  
- Reward function and terminal condition definitions  
- Access to in-game variables (kills, health, score)  
- Debug logging during training  

### `agents/dqn_model.py`
Defines the DQN neural network architecture, including convolutional layers and fully connected layers.  
Model structure follows the Atari DQN, allowing:

- Reuse across training scripts  
- Easy modification (e.g., Dueling DQN, Double DQN)  

### `agents/dqn_agent.py`
Implements the logic for the DQN agent:

- Builds the policy and target networks  
- Implements ε-greedy strategy for exploration  
- Calculates Q-values and training loss  
- Periodically updates the target network  

### `utils/replay_buffer.py`
Implements the Replay Buffer:

- Stores (state, action, reward, next_state, done) tuples  
- Supports random mini-batch sampling  
- Prevents sample correlation to stabilize training  

### `train/`
Includes training scripts and logs:

- `train_dqn.py`: Main script to configure and run training, save checkpoints, and log metrics  
- `all_losses.npy` and `reward_log.npy`: Updated every 10 episodes for visualization  

### `tests/`
Unit tests for core components:

- `test_env.py`: Tests environment reset and step  
- `test_agent.py`: Validates action selection and learning  
- `test_replaybuffer.py`: Checks buffer functionality  
- `test_dqn_model.py`: Runs inference with the trained model  

---

## 🖼️ 4. State Design
The agent uses stacked game screen images as its state input instead of numerical vectors.  
Each state is composed of 4 consecutive grayscale frames of size 84×84 (shape: `(4, 84, 84)`).  
This captures dynamic elements such as object motion and speed.

The frame processing pipeline includes:

1. Convert the Pygame screen to a NumPy array  
2. Resize the frame to 84×84  
3. Convert the frame to grayscale  
4. Stack the most recent 4 frames  

---

## 💾 Model for Inference
The trained model used for gameplay inference is located in the `checkpoints/` folder and is named `dqn_episode_1300.pth`.

