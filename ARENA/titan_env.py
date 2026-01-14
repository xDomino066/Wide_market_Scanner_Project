import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random

# --- KONFIGURACJA ---
INITIAL_BALANCE = 10_000
COMMISSION = 0.005  # 0.1% prowizji
WINDOW_SIZE = 60  # Ile dni widzi model


class TitanGymEnv(gym.Env):
    """
    Arena TRI-FUSION (Dostosowana do nagłówków użytkownika):
    1. AI Ready (Technika + Fundamenty Statyczne)
    2. Growth (Wzrosty QQ/YY)
    3. Real Market (Prawdziwe ceny do symulacji)
    """

    def __init__(self, ai_ready_path, growth_path, prices_path):
        super(TitanGymEnv, self).__init__()

        print("🏟️ Inicjalizacja Areny TITAN (Custom Headers)...")

        # 1. ŁADOWANIE PLIKÓW
        # A. AI READY (Główne dane)
        # Kolumny: Date,Open,High,Low,Close,Volume,...,NetMargin,ROE,ROA,PE_Ratio,Ticker
        print(f"   1/3 Ładowanie AI Ready: {ai_ready_path}...")
        df_ai = pd.read_csv(ai_ready_path)
        df_ai['Date'] = pd.to_datetime(df_ai['Date'])

        # B. GROWTH (Wzrosty)
        # Kolumny: Date,Ticker,Year,Quarter,Revenue_QQ,Revenue_YY...
        print(f"   2/3 Ładowanie Growth: {growth_path}...")
        df_growth = pd.read_csv(growth_path)
        df_growth['Date'] = pd.to_datetime(df_growth['Date'])

        # C. REAL PRICES (Portfel)
        # Kolumny: Date,Ticker,Raw_Open,Raw_Close
        print(f"   3/3 Ładowanie Cen Rynkowych: {prices_path}...")
        df_prices = pd.read_csv(prices_path)
        df_prices['Date'] = pd.to_datetime(df_prices['Date'])

        # 2. SCALANIE (THE BIG MERGE)
        print("   🔗 Fuzja danych...")

        # Merge 1: AI + Growth (po Ticker i Date)
        # Używamy suffixów, żeby usunąć duplikaty Year/Quarter
        df_merged = pd.merge(df_ai, df_growth, on=['Ticker', 'Date'], how='inner', suffixes=('', '_DROP'))

        # Usuwamy zduplikowane kolumny (np. Year_DROP, Quarter_DROP jeśli istnieją)
        df_merged = df_merged[[c for c in df_merged.columns if not c.endswith('_DROP')]]

        # Merge 2: + Real Prices
        self.df = pd.merge(df_merged, df_prices, on=['Ticker', 'Date'], how='inner')

        # Sortowanie (krytyczne!)
        self.df.sort_values(['Ticker', 'Date'], inplace=True)

        # 3. PRZYGOTOWANIE PAMIĘCI
        self.tickers = self.df['Ticker'].unique()
        self.grouped = self.df.groupby('Ticker')

        print(f"✅ Arena gotowa! Aktywa: {len(self.tickers)}. Wierszy: {len(self.df)}")

        # 4. DEFINICJA CECH (Co widzi AI?)
        # Musimy wyrzucić kolumny, których AI nie powinna widzieć (opisowe lub przyszłość)
        # RAW_OPEN i RAW_CLOSE to ceny do portfela, AI widzi swoje skalowane 'Open', 'Close' z pliku AI_READY

        ignore_cols = [
            'Date', 'Ticker', 'Entity', 'CIK',  # Metadane
            'Raw_Open', 'Raw_Close',  # Ceny dla portfela (nie dla AI)
            'Year', 'Quarter'  # Opcjonalnie: można zostawić, ale AI woli liczby ciągłe
        ]

        # Wszystko co zostało, to Input (RSI, PE_Ratio, Revenue_YY, Assets_QQ...)
        self.feature_cols = [c for c in self.df.columns if c not in ignore_cols]

        print(f"🧠 Inputy dla AI ({len(self.feature_cols)}):")
        print(f"   np. {self.feature_cols[:5]} ... {self.feature_cols[-5:]}")

        # Przestrzenie Gymnasium
        self.action_space = spaces.Discrete(4)  # 0=Hold, 1=Buy, 2=Sell, 3=Exit
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(WINDOW_SIZE, len(self.feature_cols)),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Pętla bezpieczeństwa: Losujemy tak długo, aż trafimy na dobrą spółkę
        # Limitujemy próby, żeby nie zawiesić programu w nieskończoność
        for _ in range(100):
            self.current_ticker = random.choice(self.tickers)
            ticker_df = self.grouped.get_group(self.current_ticker)

            # SPRAWDZENIE 1: Czy mamy wystarczająco dużo wierszy?
            # Musimy mieć 60 dni na okno + 100 dni na grę = 160 dni minimum
            if len(ticker_df) > WINDOW_SIZE + 100:
                break
        else:
            # Jeśli po 100 próbach nic nie znajdziemy (mało prawdopodobne, ale możliwe)
            # Wtedy bierzemy największą spółkę z całego datasetu "na siłę"
            print("⚠️ OSTRZEŻENIE: Trudności ze znalezieniem spółki z długą historią.")
            # Znajdź ticker z największą ilością danych
            sizes = self.grouped.size()
            best_ticker = sizes.idxmax()
            self.current_ticker = best_ticker
            ticker_df = self.grouped.get_group(best_ticker)

        # B. Dane do RAM
        self.features_data = ticker_df[self.feature_cols].values.astype(np.float32)
        self.price_data = ticker_df['Raw_Close'].values.astype(np.float32)

        if 'Raw_Open' in ticker_df.columns:
            self.open_price_data = ticker_df['Raw_Open'].values.astype(np.float32)
        else:
            self.open_price_data = self.price_data

        # C. Losuj start
        self.max_idx = len(ticker_df) - 1

        # Zabezpieczenie przed błędem randrange(60, 28)
        # Upewniamy się, że koniec przedziału jest ZAWSZE większy niż początek
        safe_end_idx = max(WINDOW_SIZE + 1, self.max_idx - 50)

        self.current_step = random.randint(WINDOW_SIZE, safe_end_idx)

        # D. Reset konta
        self.balance = INITIAL_BALANCE
        self.net_worth = INITIAL_BALANCE
        self.max_net_worth = INITIAL_BALANCE
        self.held_units = 0
        self.entry_price = 0
        self.position = 0

        return self._get_observation(), {}

    def _get_observation(self):
        # Wycinamy okno [t-60 : t]
        return self.features_data[self.current_step - WINDOW_SIZE: self.current_step]

    def step(self, action):
        # Cena na dzisiaj (Close) - do wyceny portfela "na papierze"
        current_price = self.price_data[self.current_step]

        # Cena egzekucji (Next Open) - Symulacja wejścia rano następnego dnia
        # Jeśli jesteśmy na końcu danych, bierzemy current_price
        if self.current_step + 1 < len(self.open_price_data):
            exec_price = self.open_price_data[self.current_step + 1]
        else:
            exec_price = current_price

        # 1. Wycena (Mark-to-Market po cenie Close)
        if self.position == 1:  # Long
            self.net_worth = self.held_units * current_price
        elif self.position == -1:  # Short
            # Profit = (Entry - Current) * Units
            profit = (self.entry_price - current_price) * self.held_units
            self.net_worth = self.balance + profit
        else:
            self.net_worth = self.balance

        # 2. Logika Handlu (Egzekucja po cenie OPEN następnego dnia)
        if action == 1:  # BUY
            if self.position != 1:
                self._close_position(exec_price)
                # Obliczamy ile możemy kupić (minus prowizja)
                cost = self.net_worth * (1 - COMMISSION)
                self.held_units = cost / exec_price
                self.entry_price = exec_price
                self.position = 1
                self.balance = 0  # Cała gotówka w akcjach

        elif action == 2:  # SELL (SHORT)
            if self.position != -1:
                self._close_position(exec_price)
                collateral = self.net_worth * (1 - COMMISSION)
                self.held_units = collateral / exec_price
                self.entry_price = exec_price
                self.position = -1
                self.balance = collateral

        elif action == 3:  # EXIT (CLOSE ALL)
            self._close_position(exec_price)

        # 3. Krok
        self.current_step += 1
        done = self.current_step >= self.max_idx

        # 4. Nagroda (Sortino Style)
        safe_net_worth = max(self.net_worth, 1e-6)
        # Log-return (stabilne dla PPO)
        reward = np.log(safe_net_worth / max(INITIAL_BALANCE, 1e-6)) / 100

        # Kara za Drawdown (Obsunięcie kapitału)
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        reward -= drawdown * 0.1  # Kara 10% za każdy % obsunięcia

        # Bankructwo (-50%)
        if self.net_worth < INITIAL_BALANCE * 0.5:
            done = True
            reward = -1

        return self._get_observation(), reward, done, False, {}

    def _close_position(self, price):
        if self.position == 0: return

        if self.position == 1:  # Zamknięcie Longa
            self.balance = self.held_units * price * (1 - COMMISSION)
        elif self.position == -1:  # Zamknięcie Shorta
            profit = (self.entry_price - price) * self.held_units
            self.balance = (self.balance + profit) * (1 - COMMISSION)

        self.position = 0
        self.held_units = 0
        self.net_worth = self.balance