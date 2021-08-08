from all_the_tools.config import Config as C

config = C(
    seed=42,
    env="InvertedDoublePendulumPyBulletEnv-v0",
    episodes=100000,
    log_interval=1000,
    transforms=[],
    gamma=0.99,
    entropy_weight=1e-2,
    adv_norm=False,
    grad_clip_norm=1.0,
    horizon=32,
    workers=32,
    model=C(encoder=C(type="fc", out_features=32), rnn=C(type="noop")),
    opt=C(type="adam", lr=1e-3),
)
