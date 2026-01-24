import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random

# --- KONFIGURACJA EKONOMICZNA (POPRAWIONA) ---
INITIAL_BALANCE = 10_000
COMMISSION = 0.001  # ZMIANA: 0.1% prowizji (było 0.5% - zabójstwo dla modelu)
SLIPPAGE = 0.001  # Poślizg cenowy (realizm)
WINDOW_SIZE = 60  # Ile dni widzi model
INACTIVITY_PENALTY = 0.0001  # ZMIANA: Mała kara za siedzenie w gotówce (zmusza do szukania)


class TitanGymEnv(gym.Env):
    def __init__(self, ai_ready_path, growth_path, prices_path):
        super(TitanGymEnv, self).__init__()

        print("🏟️ Inicjalizacja Areny TITAN (Wersja 2.0 - Fix Ekonomii)...")

        # 1. ŁADOWANIE DANYCH (Szybki Engine PyArrow dla wydajności)
        print(f"   1/3 Ładowanie AI Ready: {ai_ready_path}...")
        df_ai = pd.read_csv(ai_ready_path, engine="pyarrow")
        df_ai['Date'] = pd.to_datetime(df_ai['Date'])

        print(f"   2/3 Ładowanie Growth: {growth_path}...")
        df_growth = pd.read_csv(growth_path, engine="pyarrow")
        df_growth['Date'] = pd.to_datetime(df_growth['Date'])

        print(f"   3/3 Ładowanie Cen Rynkowych: {prices_path}...")
        df_prices = pd.read_csv(prices_path, engine="pyarrow")
        df_prices['Date'] = pd.to_datetime(df_prices['Date'])

        print("   🔗 Fuzja danych...")
        # Merge 1: AI + Growth
        df_merged = pd.merge(df_ai, df_growth, on=['Ticker', 'Date'], how='inner', suffixes=('', '_DROP'))
        df_merged = df_merged[[c for c in df_merged.columns if not c.endswith('_DROP')]]

        # Merge 2: + Real Prices
        self.df = pd.merge(df_merged, df_prices, on=['Ticker', 'Date'], how='inner')
        self.df.sort_values(['Ticker', 'Date'], inplace=True)

        # Optymalizacja pamięci
        float_cols = self.df.select_dtypes(include=['float64']).columns
        self.df[float_cols] = self.df[float_cols].astype(np.float32)

        self.tickers = self.df['Ticker'].unique()
        self.grouped = self.df.groupby('Ticker')

        print(f"✅ Arena gotowa! Aktywa: {len(self.tickers)}. Wierszy: {len(self.df)}")

        # DEFINICJA INPUTÓW
        ignore_cols = ['Date', 'Ticker', 'Entity', 'CIK', 'Raw_Open', 'Raw_Close', 'Year', 'Quarter']
        self.feature_cols = [c for c in self.df.columns if c not in ignore_cols]

        # PRZESTRZENIE GYM
        self.action_space = spaces.Discrete(4)  # 0=Hold, 1=Buy, 2=Short, 3=Close
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(WINDOW_SIZE, len(self.feature_cols)),
            dtype=np.float32
        )

        # Zmienne stanu
        self.current_ticker = None
        self.current_step = 0
        self.max_idx = 0
        self.balance = INITIAL_BALANCE
        self.net_worth = INITIAL_BALANCE
        self.prev_net_worth = INITIAL_BALANCE  # WAŻNE: Do liczenia nagrody krokowej
        self.held_units = 0
        self.position = 0
        self.entry_price = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Pętla bezpieczeństwa - szukamy spółki z wystarczającą historią
        for _ in range(100):
            self.current_ticker = random.choice(self.tickers)
            ticker_df = self.grouped.get_group(self.current_ticker)
            if len(ticker_df) > WINDOW_SIZE + 150:
                break
        else:
            sizes = self.grouped.size()
            self.current_ticker = sizes.idxmax()
            ticker_df = self.grouped.get_group(self.current_ticker)

        # Dane do RAM
        self.features_data = ticker_df[self.feature_cols].values.astype(np.float32)
        self.price_data = ticker_df['Raw_Close'].values.astype(np.float32)

        if 'Raw_Open' in ticker_df.columns:
            self.open_price_data = ticker_df['Raw_Open'].values.astype(np.float32)
        else:
            self.open_price_data = self.price_data

        # Reset zmiennych
        self.max_idx = len(ticker_df) - 1
        safe_end = max(WINDOW_SIZE + 1, self.max_idx - 100)
        self.current_step = random.randint(WINDOW_SIZE, safe_end)

        self.balance = INITIAL_BALANCE
        self.net_worth = INITIAL_BALANCE
        self.prev_net_worth = INITIAL_BALANCE
        self.held_units = 0
        self.entry_price = 0
        self.position = 0

        return self._get_observation(), {}

    def _get_observation(self):
        obs = self.features_data[self.current_step - WINDOW_SIZE: self.current_step]
        # Padding w razie błędu (rzadkie)
        if len(obs) < WINDOW_SIZE:
            padding = np.zeros((WINDOW_SIZE - len(obs), obs.shape[1]), dtype=np.float32)
            obs = np.vstack([padding, obs])
        return obs

    def step(self, action):
        # 1. DANE RYNKOWE
        current_price = self.price_data[self.current_step]
        # Egzekucja po cenie otwarcia następnego dnia (realizm)
        exec_price = self.open_price_data[self.current_step + 1] if (
                                                                                self.current_step + 1) <= self.max_idx else current_price

        # 2. MARK-TO-MARKET (Wycena przed ruchem)
        if self.position == 1:
            self.net_worth = self.held_units * current_price
        elif self.position == 2:  # Short
            profit = (self.entry_price - current_price) * self.held_units
            self.net_worth = self.balance + profit
        else:
            self.net_worth = self.balance

        # 3. LOGIKA HANDLU
        if action == 3:  # CLOSE
            if self.position != 0:
                # Uwzględniamy poślizg (Slippage)
                realized_price = exec_price * (1 - SLIPPAGE) if self.position == 1 else exec_price * (1 + SLIPPAGE)

                if self.position == 1:
                    gross_val = self.held_units * realized_price
                    self.balance = gross_val * (1 - COMMISSION)
                elif self.position == 2:
                    profit = (self.entry_price - realized_price) * self.held_units
                    self.balance = (self.balance + profit) * (1 - COMMISSION)

                self.held_units = 0
                self.position = 0
                self.net_worth = self.balance

        elif action == 1 and self.position == 0:  # BUY
            # Wejście z poślizgiem + prowizja
            real_exec_price = exec_price * (1 + SLIPPAGE)
            cost = self.balance * (1 - COMMISSION)
            self.held_units = cost / real_exec_price
            self.entry_price = real_exec_price
            self.position = 1
            self.balance = 0
            # Aktualizacja wyceny od razu po wejściu (uwzględnia spread)
            self.net_worth = self.held_units * current_price

        elif action == 2 and self.position == 0:  # SHORT
            real_exec_price = exec_price * (1 - SLIPPAGE)
            cost = self.balance * (1 - COMMISSION)
            self.held_units = cost / real_exec_price
            self.entry_price = real_exec_price
            self.position = 2
            # Balance w shorcie traktujemy jako depozyt

        # 4. OBLICZANIE NAGRODY (NOWY SYSTEM)
        # Step Return: Nagradzamy za zmianę kapitału w tym konkretnym kroku

        safe_prev = max(self.prev_net_worth, 1e-6)
        step_return = np.log(self.net_worth / safe_prev)

        # Skalowanie: x100 żeby wartości były czytelne dla sieci neuronowej
        reward = step_return * 100

        # Kara za bierność (jeśli nie mam pozycji)
        if self.position == 0:
            reward -= INACTIVITY_PENALTY

        # Zapamiętujemy stan na następny krok
        self.prev_net_worth = self.net_worth
        self.current_step += 1

        done = False
        # Bankructwo (-40% kapitału - ciasny stop loss globalny)
        if self.net_worth < INITIAL_BALANCE * 0.6:
            done = True
            reward = -10  # Duża kara na koniec

        if self.current_step >= self.max_idx - 1:
            done = True

        return self._get_observation(), reward, done, False, {}