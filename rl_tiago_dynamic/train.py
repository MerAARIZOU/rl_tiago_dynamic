from tiago_env import *
from stable_baselines3 import TD3, DDPG, DQN
from stable_baselines3.common.callbacks import CheckpointCallback

rclpy.init()
# Création de l'environnement
env = TiagoEnv()

# Initialisation de l'algorithme (TD3 est souvent le meilleur pour la navigation continue)
model = TD3("MlpPolicy", env, device="cuda", verbose=0, learning_rate=0.001)

checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path='./logs/models/',
    name_prefix='tiago_model'
)

# Lancement de l'apprentissage (100 000 steps par exemple)
model.learn(total_timesteps=100000, callback=checkpoint_callback)

# Sauvegarde
model.save("tiago_td3_model")
#env = TiagoEnv()
#for _ in range(100):
#    action = env.action_space.sample() # Action aléatoire
#    obs, reward, done, _, _ = env.step(action)
#    print(f"Reward: {reward}")
#    if done:
#        env.reset()