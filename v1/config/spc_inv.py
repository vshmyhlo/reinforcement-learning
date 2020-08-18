from config import Config as C

config = C(
    seed=42,
    env='SpaceInvaders-v0',
    episodes=100000,
    log_interval=100,
    transforms=[
        C(type='grayscale'),
        C(type='stack', k=4, dim=0),
        C(type='skip', k=4),
        C(type='normalize'),
    ],
    gamma=0.99,
    entropy_weight=1e-2,
    horizon=8,
    workers=32,
    model=C(
        size=128,
        encoder=C(
            type='conv',
            size=16,
            shared=True)),
    opt=C(
        type='adam',
        lr=1e-3))
