import os
import torch
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor  # <--- NOWOŚĆ: KLUCZOWY IMPORT

# --- IMPORTY Z TWOICH FOLDERÓW ---
from ARENA.titan_env import TitanGymEnv
from MODEL.titan_model import policy_kwargs

warnings.filterwarnings("ignore")

# --- KONFIGURACJA ŚCIEŻEK ---
BASE_DATA_DIR = "dataset"
AI_FILE = os.path.join(BASE_DATA_DIR, "sp500_AI_READY_FINAL.csv")
GROWTH_FILE = os.path.join(BASE_DATA_DIR, "sp500_growth_DAILY_CLEAN.csv")
PRICE_FILE = os.path.join(BASE_DATA_DIR, "real_market_prices.csv")

LOG_DIR = "./titan_logs/"
MODEL_DIR = "./titan_models/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    print("🚀 INICJALIZACJA SYSTEMU TITAN (Z MONITOREM)...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Urządzenie: {device.upper()}")
    if device == "cuda":
        print(f"   Karta: {torch.cuda.get_device_name(0)}")

    print("🏟️ Tworzenie Areny...")

    # --- TU JEST ZMIANA: DODANO MONITOR ---
    # Monitor sprawia, że TensorBoard widzi nagrody (rollout/ep_rew_mean)
    env = DummyVecEnv([lambda: Monitor(TitanGymEnv(AI_FILE, GROWTH_FILE, PRICE_FILE))])

    print("🧠 Budowanie modelu...")

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device=device,
        tensorboard_log=LOG_DIR
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=MODEL_DIR,
        name_prefix="titan_transformer"
    )

    TOTAL_STEPS = 5_000_000
    print(f"\n🏎️ START TRENINGU ({TOTAL_STEPS} kroków)...")
    print(f"   Wykresy: tensorboard --logdir {LOG_DIR}")

    try:
        model.learn(total_timesteps=TOTAL_STEPS, callback=checkpoint_callback)
        model.save("titan_final_model")
        print("✅ Trening zakończony!")
    except KeyboardInterrupt:
        print("\n🛑 Przerwano. Zapisywanie...")
        model.save("titan_interrupted_model")


if __name__ == "__main__":
    main()