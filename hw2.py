import numpy as np
import matplotlib.pyplot as plt

# =========================
# Environment Setting
# =========================
ROWS = 4
COLS = 12

START = (3, 0)
GOAL = (3, 11)

ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1)    # right
}

ACTION_SYMBOLS = {
    0: "↑",
    1: "↓",
    2: "←",
    3: "→"
}

def is_cliff(state):
    row, col = state
    return row == 3 and 1 <= col <= 10

def step(state, action):
    dr, dc = ACTIONS[action]
    row, col = state

    next_row = row + dr
    next_col = col + dc

    # Boundary check
    next_row = max(0, min(ROWS - 1, next_row))
    next_col = max(0, min(COLS - 1, next_col))

    next_state = (next_row, next_col)

    if is_cliff(next_state):
        return START, -100, False

    if next_state == GOAL:
        return next_state, -1, True

    return next_state, -1, False

def state_to_index(state):
    row, col = state
    return row * COLS + col

def epsilon_greedy(Q, state, epsilon):
    state_idx = state_to_index(state)

    if np.random.rand() < epsilon:
        return np.random.randint(4)
    else:
        return np.argmax(Q[state_idx])

# =========================
# Q-learning
# =========================
def train_q_learning(episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    Q = np.zeros((ROWS * COLS, 4))
    rewards_per_episode = []

    for ep in range(episodes):
        state = START
        done = False
        total_reward = 0

        while not done:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done = step(state, action)

            s_idx = state_to_index(state)
            ns_idx = state_to_index(next_state)

            # Q-learning update
            Q[s_idx, action] += alpha * (
                reward + gamma * np.max(Q[ns_idx]) - Q[s_idx, action]
            )

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode

# =========================
# SARSA
# =========================
def train_sarsa(episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    Q = np.zeros((ROWS * COLS, 4))
    rewards_per_episode = []

    for ep in range(episodes):
        state = START
        action = epsilon_greedy(Q, state, epsilon)
        done = False
        total_reward = 0

        while not done:
            next_state, reward, done = step(state, action)

            s_idx = state_to_index(state)
            ns_idx = state_to_index(next_state)

            if done:
                target = reward
            else:
                next_action = epsilon_greedy(Q, next_state, epsilon)
                target = reward + gamma * Q[ns_idx, next_action]

            # SARSA update
            Q[s_idx, action] += alpha * (target - Q[s_idx, action])

            state = next_state

            if not done:
                action = next_action

            total_reward += reward

        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode

# =========================
# Multiple Runs Average
# =========================
def average_runs(algorithm, runs=50, episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    all_rewards = []

    final_Q = None

    for run in range(runs):
        Q, rewards = algorithm(
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon
        )
        all_rewards.append(rewards)
        final_Q = Q

    return final_Q, np.mean(all_rewards, axis=0)

# =========================
# Print Policy
# =========================
def print_policy(Q, title):
    print("\n" + title)

    for r in range(ROWS):
        row_symbols = []

        for c in range(COLS):
            state = (r, c)

            if state == START:
                row_symbols.append("S")
            elif state == GOAL:
                row_symbols.append("G")
            elif is_cliff(state):
                row_symbols.append("C")
            else:
                best_action = np.argmax(Q[state_to_index(state)])
                row_symbols.append(ACTION_SYMBOLS[best_action])

        print(" ".join(row_symbols))

# =========================
# Main
# =========================
episodes = 500
runs = 50
alpha = 0.5
gamma = 1.0
epsilon = 0.1

Q_qlearning, qlearning_rewards = average_runs(
    train_q_learning,
    runs=runs,
    episodes=episodes,
    alpha=alpha,
    gamma=gamma,
    epsilon=epsilon
)

Q_sarsa, sarsa_rewards = average_runs(
    train_sarsa,
    runs=runs,
    episodes=episodes,
    alpha=alpha,
    gamma=gamma,
    epsilon=epsilon
)

# Plot reward curve
plt.figure(figsize=(10, 6))
plt.plot(sarsa_rewards, label="SARSA")
plt.plot(qlearning_rewards, label="Q-learning")
plt.xlabel("Episodes")
plt.ylabel("Reward Sum for Episode")
plt.title("SARSA vs Q-learning in Cliff Walking")
plt.legend()
plt.grid(True)

plt.ylim(-200, 0)
plt.xlim(0, 500)
#plt.yticks(np.arange(-100, 1, 20))
plt.show()

# Print final policies
print_policy(Q_qlearning, "Q-learning Policy")
print_policy(Q_sarsa, "SARSA Policy")