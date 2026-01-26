"""
TITAN LSTM Training Script (RecurrentPPO)
==========================================
Architektura "Czolg" - stabilna pamiec sekwencyjna zamiast Transformera.

Uruchomienie:
    python train_lstm.py              # Pelny trening (5M krokow)
    python train_lstm.py --test-run   # Sanity check (10k krokow)
"""

import os
import sys
import argparse
import torch
import warnings
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# --- IMPORTY LOKALNE ---
from ARENA.titan_env import TitanGymEnv

warnings.filterwarnings("ignore")


# ============================================================================
#                           KONFIGURACJA
# ============================================================================

# Sciezki danych
BASE_DATA_DIR = "dataset"
AI_FILE = os.path.join(BASE_DATA_DIR, "sp500_AI_READY_FINAL.csv")
GROWTH_FILE = os.path.join(BASE_DATA_DIR, "sp500_growth_DAILY_CLEAN.csv")
PRICE_FILE = os.path.join(BASE_DATA_DIR, "real_market_prices.csv")

# Sciezki wyjsciowe
LOG_DIR = "./titan_logs_lstm/"
MODEL_DIR = "./titan_models_lstm/"

# Hiperparametry LSTM (v3.1 - optymalizacja szybkosci uczenia)
LSTM_CONFIG = {
    "learning_rate": 5e-4,      # ZWIEKSZONE: szybsze uczenie
    "n_steps": 2048,            # Wiecej krokow przed aktualizacja
    "batch_size": 256,          # Wiekszy batch dla GPU utilization
    "n_epochs": 10,             # Wiecej przejsc przez dane
    "gamma": 0.99,              # Discount factor
    "gae_lambda": 0.95,         # GAE lambda
    "clip_range": 0.2,          # Wiekszy clip dla 9 akcji
    "ent_coef": 0.05,           # Wiecej eksploracji
    "vf_coef": 0.5,             # Value function coefficient
    "max_grad_norm": 0.5,       # Gradient clipping
}

# Architektura LSTM (v3.1 - mniejsza dla szybszego uczenia)
POLICY_KWARGS = {
    "lstm_hidden_size": 256,    # ZMNIEJSZONE: 512 -> 256
    "n_lstm_layers": 1,         # ZMNIEJSZONE: 2 -> 1
    "shared_lstm": False,       # Oddzielne LSTM dla actor i critic
    "enable_critic_lstm": True, # LSTM rowniez dla value function
    "net_arch": dict(
        pi=[128, 64],           # Mniejszy actor network
        vf=[128, 64]            # Mniejszy critic network
    )
}


# ============================================================================
#                           FUNKCJE POMOCNICZE
# ============================================================================

def create_env(train_mode=True):
    """Tworzy srodowisko z monitorem."""
    env = TitanGymEnv(AI_FILE, GROWTH_FILE, PRICE_FILE, train_mode=train_mode)
    return Monitor(env)


def get_device():
    """Wykrywa urzadzenie (GPU/CPU)."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        device = "cpu"
        print("[WARNING] CUDA niedostepna - trening na CPU (wolne!)")
    return device


# ============================================================================
#                           GLOWNA FUNKCJA
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TITAN LSTM Training")
    parser.add_argument("--test-run", action="store_true", 
                        help="Krotki test (10k krokow)")
    parser.add_argument("--continue-training", type=str, default=None,
                        help="Sciezka do modelu do kontynuacji treningu")
    args = parser.parse_args()

    # Ustawienia
    total_steps = 10_000 if args.test_run else 5_000_000
    save_freq = 5_000 if args.test_run else 50_000

    print("=" * 60)
    print("TITAN LSTM (Czolg) - RecurrentPPO")
    print("=" * 60)
    
    # Katalogi
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Urzadzenie
    device = get_device()

    # Srodowisko
    print("\n[ARENA] Inicjalizacja Areny...")
    env = DummyVecEnv([create_env])

    # Model
    if args.continue_training:
        print(f"\n[LOAD] Wczytywanie modelu: {args.continue_training}")
        model = RecurrentPPO.load(
            args.continue_training,
            env=env,
            device=device,
            tensorboard_log=LOG_DIR
        )
        print("[OK] Model wczytany!")
    else:
        print("\n[BUILD] Budowanie modelu LSTM...")
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            device=device,
            tensorboard_log=LOG_DIR,
            policy_kwargs=POLICY_KWARGS,
            **LSTM_CONFIG
        )

    # Callback do zapisywania checkpointow
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=MODEL_DIR,
        name_prefix="titan_lstm"
    )

    # Podsumowanie
    print("\n" + "-" * 40)
    print("KONFIGURACJA TRENINGU:")
    print(f"   - Krokow: {total_steps:,}")
    print(f"   - Learning Rate: {LSTM_CONFIG['learning_rate']}")
    print(f"   - LSTM Hidden Size: {POLICY_KWARGS['lstm_hidden_size']}")
    print(f"   - Batch Size: {LSTM_CONFIG['batch_size']}")
    print(f"   - Entropy Coef: {LSTM_CONFIG['ent_coef']}")
    print("-" * 40)
    print(f"[TENSORBOARD] tensorboard --logdir {LOG_DIR}")
    print("-" * 40 + "\n")

    # Trening
    try:
        print("[START] START TRENINGU...")
        model.learn(
            total_timesteps=total_steps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Zapis koncowy
        final_path = os.path.join(MODEL_DIR, "titan_lstm_final")
        model.save(final_path)
        print(f"\n[DONE] Trening zakonczony! Model zapisany: {final_path}")
        
    except KeyboardInterrupt:
        print("\n\n[STOP] Przerwano przez uzytkownika.")
        interrupted_path = os.path.join(MODEL_DIR, "titan_lstm_interrupted")
        model.save(interrupted_path)
        print(f"[SAVE] Model zapisany: {interrupted_path}")
        sys.exit(0)


if __name__ == "__main__":
    main()
