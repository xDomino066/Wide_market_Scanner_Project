import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random

# =============================================================================
#                    TITAN ENVIRONMENT v3.0 (SOTA)
# =============================================================================
# Features:
# - 9 akcji (position sizing + trimming)
# - Continuous Position Rewards
# - Trade Outcome Bonus v2
# - Train/Test Split
# - Turnover Penalty
# - Extended Observation Space
# =============================================================================

# --- KONFIGURACJA EKONOMICZNA ---
INITIAL_BALANCE = 10_000
COMMISSION = 0.001      # 0.1% prowizji
SLIPPAGE = 0.001        # 0.1% poslizg cenowy
WINDOW_SIZE = 60        # Ile dni widzi model

# --- KONFIGURACJA REWARD ---
POSITION_REWARD_SCALE = 0.2   # Skala continuous rewards (0.5% ruch -> 0.1 reward)
INACTIVITY_PENALTY = 0.01     # Kara za biernosc
OVERTRADING_THRESHOLD = 5     # Po ilu transakcjach kara za overtrading


class TitanGymEnv(gym.Env):
    """
    TITAN SOTA Environment v3.0
    
    Action Space (9 akcji):
        0 = HOLD (nic nie rob)
        1 = BUY 25% kapitalu
        2 = BUY 50% kapitalu  
        3 = BUY 100% kapitalu
        4 = SHORT 25% kapitalu
        5 = SHORT 50% kapitalu
        6 = SHORT 100% kapitalu
        7 = CLOSE 50% pozycji (TRIM)
        8 = CLOSE 100% pozycji (FULL EXIT)
    """
    
    def __init__(self, ai_ready_path, growth_path, prices_path, 
                 train_mode=True, train_ratio=0.8, seed=42):
        super(TitanGymEnv, self).__init__()
        
        mode_str = "TRAIN" if train_mode else "TEST"
        print(f"[ARENA] Inicjalizacja TITAN v3.0 SOTA ({mode_str} mode)...")
        
        self.train_mode = train_mode
        
        # 1. LADOWANIE DANYCH
        print(f"   1/3 Ladowanie AI Ready: {ai_ready_path}...")
        df_ai = pd.read_csv(ai_ready_path, engine="pyarrow")
        df_ai['Date'] = pd.to_datetime(df_ai['Date'])
        
        print(f"   2/3 Ladowanie Growth: {growth_path}...")
        df_growth = pd.read_csv(growth_path, engine="pyarrow")
        df_growth['Date'] = pd.to_datetime(df_growth['Date'])
        
        print(f"   3/3 Ladowanie Cen Rynkowych: {prices_path}...")
        df_prices = pd.read_csv(prices_path, engine="pyarrow")
        df_prices['Date'] = pd.to_datetime(df_prices['Date'])
        
        print("   [MERGE] Fuzja danych...")
        df_merged = pd.merge(df_ai, df_growth, on=['Ticker', 'Date'], 
                            how='inner', suffixes=('', '_DROP'))
        df_merged = df_merged[[c for c in df_merged.columns if not c.endswith('_DROP')]]
        
        self.df = pd.merge(df_merged, df_prices, on=['Ticker', 'Date'], how='inner')
        self.df.sort_values(['Ticker', 'Date'], inplace=True)
        
        # Optymalizacja pamieci
        float_cols = self.df.select_dtypes(include=['float64']).columns
        self.df[float_cols] = self.df[float_cols].astype(np.float32)
        
        # --- TRAIN/TEST SPLIT ---
        all_tickers = self.df['Ticker'].unique()
        np.random.seed(seed)
        shuffled_tickers = np.random.permutation(all_tickers)
        
        split_idx = int(len(shuffled_tickers) * train_ratio)
        
        if train_mode:
            self.tickers = shuffled_tickers[:split_idx]
        else:
            self.tickers = shuffled_tickers[split_idx:]
        
        # Filtruj df do wybranych tickerow
        self.df = self.df[self.df['Ticker'].isin(self.tickers)]
        self.grouped = self.df.groupby('Ticker')
        
        print(f"[OK] Arena gotowa! Spolek: {len(self.tickers)} ({mode_str}). Wierszy: {len(self.df)}")
        
        # DEFINICJA INPUTOW
        ignore_cols = ['Date', 'Ticker', 'Entity', 'CIK', 'Raw_Open', 'Raw_Close', 'Year', 'Quarter']
        self.feature_cols = [c for c in self.df.columns if c not in ignore_cols]
        self.n_features = len(self.feature_cols)
        
        # --- PRZESTRZENIE GYM ---
        # 9 akcji: position sizing + trimming
        self.action_space = spaces.Discrete(9)
        
        # Obserwacja: window + dodatkowe meta-features
        # Meta features: [position, position_pct, pnl_pct, trades_count, hold_time]
        self.n_meta_features = 5
        obs_shape = (WINDOW_SIZE * self.n_features + self.n_meta_features,)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )
        
        # Zmienne stanu
        self._init_state_variables()
    
    def _init_state_variables(self):
        """Inicjalizacja/reset zmiennych stanu."""
        self.current_ticker = None
        self.current_step = 0
        self.max_idx = 0
        self.balance = INITIAL_BALANCE
        self.net_worth = INITIAL_BALANCE
        self.prev_net_worth = INITIAL_BALANCE
        self.prev_price = 0
        self.entry_net_worth = INITIAL_BALANCE
        self.held_units = 0
        self.position = 0           # 0=flat, 1=long, 2=short
        self.position_pct = 0.0     # Jaki % kapitalu w pozycji
        self.entry_price = 0
        self.hold_time = 0          # Ile krokow trzymamy pozycje
        self.trades_this_episode = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset zmiennych stanu NAJPIERW
        self._init_state_variables()
        
        # Szukamy spolki z wystarczajaca historia
        for _ in range(100):
            self.current_ticker = random.choice(list(self.tickers))
            ticker_df = self.grouped.get_group(self.current_ticker)
            if len(ticker_df) > WINDOW_SIZE + 150:
                break
        else:
            sizes = self.grouped.size()
            valid_tickers = sizes[sizes > WINDOW_SIZE + 150]
            if len(valid_tickers) > 0:
                self.current_ticker = valid_tickers.idxmax()
            else:
                self.current_ticker = sizes.idxmax()
            ticker_df = self.grouped.get_group(self.current_ticker)
        
        # Dane do RAM
        self.features_data = ticker_df[self.feature_cols].values.astype(np.float32)
        self.price_data = ticker_df['Raw_Close'].values.astype(np.float32)
        
        if 'Raw_Open' in ticker_df.columns:
            self.open_price_data = ticker_df['Raw_Open'].values.astype(np.float32)
        else:
            self.open_price_data = self.price_data
        
        # Ustaw max_idx i current_step PO init_state_variables
        self.max_idx = len(ticker_df) - 1
        safe_end = max(WINDOW_SIZE + 1, self.max_idx - 100)
        self.current_step = random.randint(WINDOW_SIZE, safe_end)
        self.prev_price = self.price_data[self.current_step]
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Zwraca obserwacje: features window + meta features."""
        # Glowne features (window)
        obs = self.features_data[self.current_step - WINDOW_SIZE: self.current_step]
        
        if len(obs) < WINDOW_SIZE:
            padding = np.zeros((WINDOW_SIZE - len(obs), self.n_features), dtype=np.float32)
            obs = np.vstack([padding, obs])
        
        # Flatten window
        obs_flat = obs.flatten()
        
        # Meta features (znormalizowane)
        pnl_pct = (self.net_worth - INITIAL_BALANCE) / INITIAL_BALANCE
        
        meta_features = np.array([
            self.position / 2.0,                    # Position: 0, 0.5, 1
            self.position_pct,                      # % kapitalu w pozycji
            np.clip(pnl_pct, -1, 1),               # PnL % (clip dla stabilnosci)
            min(self.trades_this_episode / 10.0, 1.0),  # Trades count (norm)
            min(self.hold_time / 20.0, 1.0)        # Hold time (norm)
        ], dtype=np.float32)
        
        return np.concatenate([obs_flat, meta_features])
    
    def step(self, action):
        """
        Wykonuje krok w srodowisku.
        
        Actions:
            0 = HOLD
            1 = BUY 25%
            2 = BUY 50%
            3 = BUY 100%
            4 = SHORT 25%
            5 = SHORT 50%
            6 = SHORT 100%
            7 = CLOSE 50% (TRIM)
            8 = CLOSE 100% (FULL EXIT)
        """
        current_price = self.price_data[self.current_step]
        exec_price = self.open_price_data[min(self.current_step + 1, self.max_idx)]
        
        # --- MARK-TO-MARKET ---
        if self.position == 1:  # Long
            self.net_worth = self.balance + self.held_units * current_price
        elif self.position == 2:  # Short
            profit = (self.entry_price - current_price) * self.held_units
            self.net_worth = self.balance + profit
        else:
            self.net_worth = self.balance
        
        # --- CONTINUOUS POSITION REWARD ---
        position_reward = 0.0
        if self.position != 0 and self.prev_price > 0:
            price_change_pct = (current_price - self.prev_price) / self.prev_price * 100
            
            if self.position == 1:  # Long: w gore = dobrze
                position_reward = price_change_pct * POSITION_REWARD_SCALE
            elif self.position == 2:  # Short: w dol = dobrze
                position_reward = -price_change_pct * POSITION_REWARD_SCALE
        
        # --- LOGIKA HANDLU ---
        trade_bonus = 0.0
        previous_position = self.position
        
        # CLOSE actions (7, 8)
        if action == 7 and self.position != 0:  # TRIM 50%
            trade_bonus = self._close_position(exec_price, trim_pct=0.5)
            
        elif action == 8 and self.position != 0:  # FULL EXIT
            trade_bonus = self._close_position(exec_price, trim_pct=1.0)
        
        # BUY actions (1, 2, 3)
        elif action in [1, 2, 3] and self.position != 1:
            size_pct = {1: 0.25, 2: 0.50, 3: 1.0}[action]
            
            # Jezeli mamy short, najpierw zamknij
            if self.position == 2:
                trade_bonus = self._close_position(exec_price, trim_pct=1.0)
            
            self._open_position(exec_price, position_type=1, size_pct=size_pct)
            self.trades_this_episode += 1
        
        # SHORT actions (4, 5, 6)
        elif action in [4, 5, 6] and self.position != 2:
            size_pct = {4: 0.25, 5: 0.50, 6: 1.0}[action]
            
            # Jezeli mamy long, najpierw zamknij
            if self.position == 1:
                trade_bonus = self._close_position(exec_price, trim_pct=1.0)
            
            self._open_position(exec_price, position_type=2, size_pct=size_pct)
            self.trades_this_episode += 1
        
        # Update hold time
        if self.position != 0:
            self.hold_time += 1
        else:
            self.hold_time = 0
        
        # --- OBLICZANIE NAGRODY ---
        reward = 0.0
        
        # 1. Continuous position reward (glowny sygnal)
        reward += position_reward
        
        # 2. Trade outcome bonus
        reward += trade_bonus
        
        # 3. Kara za biernosc
        if self.position == 0:
            reward -= INACTIVITY_PENALTY
        
        # 4. Kara za overtrading
        if self.trades_this_episode > OVERTRADING_THRESHOLD:
            overtrading_penalty = 0.02 * (self.trades_this_episode - OVERTRADING_THRESHOLD)
            reward -= overtrading_penalty
        
        # --- UPDATE STATE ---
        self.prev_net_worth = self.net_worth
        self.prev_price = current_price
        self.current_step += 1
        
        # --- TERMINAL CONDITIONS ---
        done = False
        truncated = False
        
        # Bankructwo
        if self.net_worth < INITIAL_BALANCE * 0.5:
            done = True
            reward = -20.0
        
        # Koniec danych
        if self.current_step >= self.max_idx - 1:
            done = True
        
        return self._get_observation(), reward, done, truncated, {}
    
    def _open_position(self, exec_price, position_type, size_pct):
        """Otwiera pozycje (long lub short)."""
        if self.balance <= 0:
            return  # Brak srodkow
            
        if position_type == 1:  # LONG
            real_exec_price = exec_price * (1 + SLIPPAGE)
            invest_amount = self.balance * size_pct
            cost_after_commission = invest_amount * (1 - COMMISSION)
            units = cost_after_commission / real_exec_price
            
            self.held_units += units
            self.balance -= invest_amount  # Odejmij uzyty kapital
            self.entry_price = real_exec_price
            self.entry_net_worth = self.net_worth
            self.position = 1
            self.position_pct = size_pct
            self.hold_time = 0
            
        elif position_type == 2:  # SHORT
            real_exec_price = exec_price * (1 - SLIPPAGE)
            invest_amount = self.balance * size_pct
            cost_after_commission = invest_amount * (1 - COMMISSION)
            units = cost_after_commission / real_exec_price
            
            self.held_units += units
            # W shorcie balance zostaje jako margin (nie odejmujemy)
            self.entry_price = real_exec_price
            self.entry_net_worth = self.net_worth
            self.position = 2
            self.position_pct = size_pct
            self.hold_time = 0
    
    def _close_position(self, exec_price, trim_pct=1.0):
        """Zamyka pozycje (calkowicie lub czesciowo)."""
        if self.position == 0 or self.held_units == 0:
            return 0.0
        
        units_to_close = self.held_units * trim_pct
        
        if self.position == 1:  # LONG
            realized_price = exec_price * (1 - SLIPPAGE)
            proceeds = units_to_close * realized_price * (1 - COMMISSION)
            self.balance += proceeds
            
        elif self.position == 2:  # SHORT
            realized_price = exec_price * (1 + SLIPPAGE)
            profit = (self.entry_price - realized_price) * units_to_close
            self.balance += profit * (1 - COMMISSION)
        
        self.held_units -= units_to_close
        
        # Trade outcome bonus
        trade_return = (self.balance + self.held_units * exec_price - self.entry_net_worth) / max(self.entry_net_worth, 1e-6)
        
        trade_bonus = 0.0
        if trim_pct == 1.0:  # Full exit - pelna ocena
            if trade_return > 0.10:
                trade_bonus = 15.0
            elif trade_return > 0.05:
                trade_bonus = 8.0
            elif trade_return > 0.02:
                trade_bonus = 3.0
            elif trade_return > 0:
                trade_bonus = 1.0
            elif trade_return > -0.02:
                trade_bonus = -0.5
            elif trade_return > -0.05:
                trade_bonus = -2.0
            else:
                trade_bonus = -5.0
        else:  # Partial exit (trim) - mniejszy bonus
            if trade_return > 0:
                trade_bonus = 0.5
        
        # Jezeli zamknelismy wszystko
        if self.held_units <= 0:
            self.held_units = 0
            self.position = 0
            self.position_pct = 0.0
            self.entry_price = 0
        else:
            self.position_pct = self.held_units * exec_price / max(self.net_worth, 1e-6)
        
        self.net_worth = self.balance + self.held_units * exec_price
        
        return trade_bonus