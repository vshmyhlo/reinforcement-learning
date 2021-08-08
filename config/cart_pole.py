from all_the_tools.config import Config as C

config = C(
    seed=42,
    env="CartPole-v1",
    episodes=10000,
    log_interval=100,
    transforms=[],
    gamma=0.99,
    entropy_weight=1e-2,
    horizon=8,
    workers=32,
    model=C(encoder=C(type="fc", out_features=32)),
    opt=C(type="adam", lr=1e-3),
)
