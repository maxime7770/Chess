import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

from src.chess_engine import GameState, Move
from src.minmax_agent import get_best_move as get_best_move_minmax


def _print_board_lines_debug(board):
    """
    Helper function to print the board state for debugging.
    """
    for row in board:
        print(" ".join(row))
    print()

# ========== 1) Neural Network Definition ==========

class DQNNetwork(nn.Module):
    """
    A simple feed-forward network that takes the 64-element board representation
    and outputs Q-values for all 4096 possible moves (encode_move gives 0..4095).
    """
    def __init__(self, input_dim=64, hidden_dim=256, output_dim=64*64):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)  # or InstanceNorm1d
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.norm2(out)

        out += identity
        out = F.relu(out)
        return out

class FinalMLPNetwork(nn.Module):
    """
    A final MLP-based DQN network for 64-element board input -> 4096 action Q-values.
    Uses multiple residual blocks, batch norm, and dropout.
    """
    def __init__(
        self,
        input_dim=64,
        hidden_dim=256,
        output_dim=64*64,
        num_res_blocks=2,
        dropout=0.2
    ):
        super(FinalMLPNetwork, self).__init__()

        # Initial projection from 64 -> hidden_dim
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        # A sequence of residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_res_blocks)
        ])

        # Final output layer to produce Q-values
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x shape: (batch_size, 64)
        Returns: Q-values of shape (batch_size, 4096)
        """
        # Initial linear + ReLU
        x = self.input_fc(x)
        x = F.relu(x)

        # Pass through the residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Final linear -> Q-values
        x = self.output_fc(x)
        return x
    


# ========== 2) Replay Memory ==========

class ReplayMemory:
    """
    Simple replay buffer to store experiences for DQN training.
    """
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), 
                np.array(action), 
                np.array(reward, dtype=np.float32), 
                np.array(next_state), 
                np.array(done, dtype=np.float32))
    
    def __len__(self):
        return len(self.memory)


# ========== 3) DQN Agent ==========

class DQNAgent:
    """
    DQN agent for chess. Uses a neural network to approximate Q-values for all possible actions.
    
    - state_size = 64 (flattened board)
    - action_size = 4096 (any start/end square pairs)
    """
    def __init__(
        self, 
        state_size=64, 
        action_size=64*64, 
        hidden_dim=256,
        lr=1e-3, 
        gamma=0.99, 
        epsilon_start=1.0, 
        epsilon_end=0.1,
        epsilon_decay=1_000_000, 
        batch_size=64, 
        target_update_freq=100, 
        replay_capacity=50_000,
    ):
        # Basic parameters
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr

        # Epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay  # Steps to go from start to end
        self.epsilon_step = 0

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_steps = 0

        # DQN networks
        # self.policy_net = DQNNetwork(state_size, hidden_dim, action_size)
        # self.target_net = DQNNetwork(state_size, hidden_dim, action_size)
        self.policy_net = FinalMLPNetwork(input_dim=state_size, hidden_dim=hidden_dim, output_dim=action_size)
        self.target_net = FinalMLPNetwork(input_dim=state_size, hidden_dim=hidden_dim, output_dim=action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayMemory(capacity=replay_capacity)

        # Device (CPU or GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

    def select_action(self, state_rep, valid_actions):
        """
        Epsilon-greedy selection. 
        - state_rep is a numpy array shape (64,).
        - valid_actions is a list of valid move indices from GameState.
        """
        # Epsilon scheduling
        self.epsilon_step += 1
        self.epsilon = max(
            self.epsilon_end, 
            1.0 - (self.epsilon_step * (1.0 - self.epsilon_end) / self.epsilon_decay)
        )

        # With probability epsilon, pick a random valid move
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Otherwise, pick the best Q-value among valid actions
        state_tensor = torch.FloatTensor(state_rep).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)  # shape: (1, 4096)
        q_values = q_values.cpu().numpy().flatten()

        # Filter out invalid actions by setting them to a very negative number
        # so they won't be chosen by argmax
        masked_q_values = np.full(self.action_size, -1e10, dtype=np.float32)
        masked_q_values[valid_actions] = q_values[valid_actions]

        best_action = int(np.argmax(masked_q_values))
        return best_action

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.
        """
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        """
        One gradient update on a sampled mini-batch from replay memory.
        """
        if len(self.memory) < self.batch_size:
            return  # Not enough samples

        self.train_steps += 1
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        # Convert to tensors
        state_t = torch.FloatTensor(state).to(self.device)
        action_t = torch.LongTensor(action).to(self.device)
        reward_t = torch.FloatTensor(reward).to(self.device)
        next_state_t = torch.FloatTensor(next_state).to(self.device)
        done_t = torch.FloatTensor(done).to(self.device)

        # Compute current Q(s, a)
        # policy_net(state) -> (batch_size, 4096)
        q_values = self.policy_net(state_t)
        # Gather Q-values for the chosen actions
        q_values = q_values.gather(1, action_t.unsqueeze(1)).squeeze(1)

        # Compute target Q-values for next state
        with torch.no_grad():
            # Double DQN approach:
            # next_actions = argmax(policy_net(next_state))
            next_q_values_policy = self.policy_net(next_state_t)
            next_actions = torch.argmax(next_q_values_policy, dim=1)
            # Use the target network to get Q-values for those actions
            next_q_values_target = self.target_net(next_state_t)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            target_q = reward_t + (1 - done_t) * self.gamma * next_q_values

        # Compute loss (Huber / MSE)
        loss = nn.functional.smooth_l1_loss(q_values, target_q)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update target network
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_reward(self, game_state, previous_board=None):
        """
        Return a shaped reward for the transition from `previous_board` to
        `game_state.board`.

        The 'previous_board' is the board position just before the current move was made.
        This function is typically called AFTER a move is made in training,
        so that we can see which piece was captured, if any.

        Shaped reward components:
        1) +1 if opponent is checkmated (the agent delivered mate).
        2) -1 if agent is checkmated.
        3)  0 if stalemate.
        4) +/- for material changes between previous and current board.
            E.g. capturing an enemy queen is +9, losing your queen is -9, etc.
        5) Optional: +0.01 if you put your opponent in check.
        6) Optional: -0.001 per move made to avoid extremely long games (small penalty).
        """

        # --- 1) Terminal rewards ---
        if game_state.is_checkmate():
            if game_state.white_to_move:
                # White to move => White is checkmated => Black wins => +1 for Black
                return +1
            else:
                # Black to move => Black is checkmated => Black loses => -1
                return -1

        if game_state.is_stalemate():
            # Stalemate => 0 reward
            return 0

        # If we’re still in a non-terminal state, shape the reward using additional signals.

        # --- 2) Material changes ---
        # We need to compare previous_board to current board. 
        # If no previous_board is given (e.g., first move), skip.
        material_reward = 0
        if previous_board is not None:
            material_reward = self.get_material_delta(previous_board, game_state.board)

        # --- 3) Optional: Check detection bonus ---
        # A small bonus if the move just delivered a check on the opponent’s king
        # (you could detect that by verifying the opponent's king is in check).
        check_bonus = 0
        if self.opponent_in_check(game_state):
            check_bonus = 0.01

        # --- 4) Optional: Small negative reward per move to discourage dragging the game
        # e.g. -0.001
        move_penalty = -0.001

        reward = material_reward + check_bonus + move_penalty
        return reward

    def get_material_delta(self, old_board, new_board):
        """
        Compare material in old_board vs. new_board from the agent's perspective.
        Now we assume the agent is BLACK, so capturing White's piece is positive, 
        losing Black's piece is negative.
        """
        piece_values = {'p': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9}
        
        old_score = 0
        new_score = 0

        for r in range(8):
            for c in range(8):
                piece_old = old_board[r][c]
                piece_new = new_board[r][c]
                
                # For the "old" board piece
                if piece_old != '--':
                    # If the piece is Black => sign=+1; if White => sign=-1
                    sign = 1 if piece_old[0] == 'b' else -1
                    val  = piece_values.get(piece_old[1], 0)
                    old_score += sign * val
                
                # For the "new" board piece
                if piece_new != '--':
                    sign = 1 if piece_new[0] == 'b' else -1
                    val  = piece_values.get(piece_new[1], 0)
                    new_score += sign * val

        # From BLACK's perspective, if new_score > old_score => black gained material => positive
        # Scale by 0.05 (or any factor you like)
        return (new_score - old_score) * 0.05

    def opponent_in_check(self, game_state):
        """
        Check if the side that just got the move done on them is in check.
        If the agent is white and we just made a move, is black's king in check?
        We'll do a quick approximation:
        """
        # If your 'pins_and_checks()' sets self.in_check correctly for the side to move,
        # you might invert game_state.white_to_move to see if the OPPONENT is in check.
        opponent_to_move = not game_state.white_to_move
        if opponent_to_move:
            # If it's the opponent's turn, 'game_state.in_check' is from the opponent's perspective
            # but typically 'in_check' is updated in get_valid_moves() or pins_and_checks() for the *current* side.
            # So you might do something like manually call:
            _ = game_state.get_valid_moves()  # This updates in_check for 'white_to_move'
            return game_state.in_check
        return False

    def train_one_game(self, max_moves=200, use_random=True, minmax_depth=1):
        """
        Train with the agent controlling White only.
        Black's moves are random in this example.
        """
        game_state = GameState()
        game_state.reset()
        game_state.white_to_move = True
        done = False
        move_count = 0
        previous_board = None

        while not done and move_count < max_moves:
            move_count += 1
            # ----- WHITE (Agent) Turn -----
            if not game_state.white_to_move:
                # Agent chooses an action
                state_rep = game_state.get_state_representation()
                valid_actions = game_state.get_valid_move_indices()

                # No moves => checkmate or stalemate for White
                if not valid_actions:
                    reward = self.get_reward(game_state, previous_board)
                    self.store_transition(state_rep, 0, reward, state_rep, True)
                    done = True
                    break

                # Choose action with epsilon-greedy
                action = self.select_action(state_rep, valid_actions)

                # Make a copy of the board
                previous_board = [row[:] for row in game_state.board]

                # Execute the move
                move_obj = game_state.get_move_from_action(action)
                game_state.make_move(move_obj)

                # Next state
                next_state_rep = game_state.get_state_representation()

                # Shaped reward
                reward = self.get_reward(game_state, previous_board)

                # Store transition
                self.store_transition(state_rep, action, reward, next_state_rep, False)

                # Check if game ended (checkmate/stalemate)
                if game_state.is_checkmate() or game_state.is_stalemate():
                    done = True
                    self.store_transition(state_rep, action, reward, next_state_rep, True)
                
                # Train on mini-batch
                self.update()

            # ----- BLACK (Opponent) Turn -----
            else:
                # Example: random move for Black
                valid_actions = game_state.get_valid_move_indices()
                if not valid_actions:
                    # Means Black is checkmated or stalemated => from White's perspective, that's a +1 or 0
                    # But we're not storing transitions for Black moves here, 
                    # so we just check if the game is done:
                    done = True
                    break

                # opponent = minmax_agent
                if use_random:
                    black_action = game_state.get_move_from_action(random.choice(valid_actions))
                else:
                    black_action = get_best_move_minmax(game_state, depth=minmax_depth)
                #move_obj = game_state.get_move_from_action(black_action)
                game_state.make_move(black_action)

            # Flip turn at the end of the loop
            #game_state.white_to_move = not game_state.white_to_move

        return reward

    def train(self, num_episodes=1000, random_fraction=0.2, minmax_depths_increment=500):
        """
        Train for a number of episodes (self-play).
        """
        num_random = int(random_fraction * num_episodes)
        for episode in range(num_episodes):
            if episode < num_random:
                # Random play for the first few episodes
                ep_reward = self.train_one_game(use_random=True)
            else:
                minmax_depth = min(1 + ((episode - num_random) // minmax_depths_increment), 6)
                ep_reward = self.train_one_game(minmax_depth=minmax_depth, use_random=False)
            if (episode + 1) % 10 == 0:
                print(f"[Episode {episode+1}/{num_episodes}] Reward: {ep_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        torch.save(self.policy_net.state_dict(), "dqn_policy.pth")
        print("Model saved to dqn_policy.pth")

# ========== 4) Global Agent + Helper for Chess Main ==========

# Define parameters
LEARNING_RATE = 2e-5
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 500_000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 1000
REPLAY_CAPACITY = 50_000


# You can instantiate a global agent here or dynamically in your code
dqn_agent = DQNAgent(
    lr=LEARNING_RATE,
    gamma=GAMMA,
    epsilon_start=EPSILON_START,
    epsilon_end=EPSILON_END,
    epsilon_decay=EPSILON_DECAY,
    batch_size=BATCH_SIZE,
    target_update_freq=TARGET_UPDATE_FREQ,
    replay_capacity=REPLAY_CAPACITY
)

def get_best_move(game_state, dqn_agent):
    """
    A helper function for your main chess program to get the next move from the DQNAgent.
    This is analogous to your old minimax 'get_best_move' usage.
    """
    state_rep = game_state.get_state_representation()
    valid_actions = game_state.get_valid_move_indices()
    if not valid_actions:
        return None  # No moves => checkmate or stalemate

    action = dqn_agent.select_action(state_rep, valid_actions)
    move_obj = game_state.get_move_from_action(action)
    return move_obj


# ========== 5) Training the Agent ==========
if __name__ == "__main__":
    dqn_agent.train(num_episodes=6000, random_fraction=0.3, minmax_depths_increment=2000)