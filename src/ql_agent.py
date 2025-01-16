import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

from src.chess_engine import GameState, Move
from src.minmax_agent import get_best_move as get_best_move_minmax

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
        target_update_freq=1000, 
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
        self.policy_net = DQNNetwork(state_size, hidden_dim, action_size)
        self.target_net = DQNNetwork(state_size, hidden_dim, action_size)
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
            # By default, is_checkmate() returns True if the *current side to move* is in check with no moves.
            # If the agent is controlling the side that just *moved*, we need to figure out who actually got checkmated.
            #
            # If 'white_to_move' is True after the move, that means it's White's turn => White is the one facing checkmate => -1 for White.
            # If your agent is always White, that means your agent lost => -1
            # If your agent is always Black, that means your agent *won* => +1.
            #
            # For self-play, you can track which side delivered mate. Let's assume your code sees "the side about to move is in checkmate => that side lost."
            if game_state.white_to_move:
                # White is checkmated => if your agent was White, that's -1 for the agent.
                return -1
            else:
                # Black is checkmated => if your agent was White, that's +1 for the agent.
                return +1

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
        For instance, +9 if we captured an enemy queen, -5 if we lost a rook, etc.
        
        If you're controlling white only, 'agent_color'='w'; 
        or if black, 'agent_color'='b'. For self-play, you might just compute a 
        net difference for the side that moved last. 
        """
        # Example piece values:
        piece_values = {'p': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9}
        
        # In a more sophisticated approach, you'd track which side is which 
        # (white or black). For brevity, assume the agent is always "white" 
        # so capturing black's piece = positive, losing white's piece = negative.

        # Count material in old_board and new_board.
        old_score = 0
        new_score = 0

        for r in range(8):
            for c in range(8):
                piece_old = old_board[r][c]
                piece_new = new_board[r][c]
                
                # If there's a piece (e.g. 'wQ', 'bN', etc.)
                if piece_old != '--':
                    sign = 1 if piece_old[0] == 'w' else -1
                    val  = piece_values.get(piece_old[1], 0)
                    old_score += sign * val
                
                if piece_new != '--':
                    sign = 1 if piece_new[0] == 'w' else -1
                    val  = piece_values.get(piece_new[1], 0)
                    new_score += sign * val

        # From white's perspective, if old_score < new_score => we gained material => positive reward
        # If old_score > new_score => we lost material => negative reward
        return (new_score - old_score) * 0.05  # scale factor if you want smaller increments

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

    def train_one_game(self, max_moves=200):
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
            if game_state.white_to_move:
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
                black_action = get_best_move_minmax(game_state, depth=2)
                #move_obj = game_state.get_move_from_action(black_action)
                game_state.make_move(black_action)

            # Flip turn at the end of the loop
            #game_state.white_to_move = not game_state.white_to_move

        return reward

    def train(self, num_episodes=1000):
        """
        Train for a number of episodes (self-play).
        """
        for episode in range(num_episodes):
            ep_reward = self.train_one_game()
            if (episode + 1) % 10 == 0:
                print(f"[Episode {episode+1}/{num_episodes}] Reward: {ep_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        torch.save(self.policy_net.state_dict(), "dqn_policy.pth")
        print("Model saved to dqn_policy.pth")

# ========== 4) Global Agent + Helper for Chess Main ==========

# You can instantiate a global agent here or dynamically in your code
dqn_agent = DQNAgent()

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
    dqn_agent.train(num_episodes=5000)