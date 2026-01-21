Titan (Alpha Version)
Titan is an advanced, experimental stock scanning system designed for the S&P 500. It leverages Deep Reinforcement Learning to identify potential market opportunities by processing a complex fusion of technical indicators and fundamental growth metrics.

Key Features:

Tri-Fusion Data System: Merges technical analysis, fundamental statistics, and quarterly/yearly growth metrics into a unified dataset prepared by myself.

Decision Engine: Powered by the Proximal Policy Optimization (PPO) algorithm via Stable Baselines3.

Transformer Architecture: Implements a custom Transformer Encoder with positional encoding as the feature extractor, allowing the model to capture temporal dependencies more effectively than traditional LSTM architectures.

Simulation Environment: Runs on a custom Gymnasium-compatible environment (TitanGymEnv) that simulates real market mechanics, including order execution, commission costs, and risk management.

Disclaimer: This project is currently in the Alpha stage and is intended for educational and research purposes only. It should not be used for real financial trading.
