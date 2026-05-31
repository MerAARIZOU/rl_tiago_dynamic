import rclpy
from tiago_env import TiagoEnv
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from launch_train import TiagoDashboardCallback # On réutilise votre callback d'affichage

rclpy.init()

# 1. On recrée l'environnement propre (Gazebo doit être relancé en parallèle)
env = Monitor(TiagoEnv(), filename="./logs/monitor_resume.csv")

# 2. CHEMIN DU MEILLEUR CHECKPOINT (Modifiez le nombre selon votre dossier)
# Exemple : si le plus haut est 15000 :
checkpoint_path = "./logs/models/tiago_model_30000_steps.zip" 

print(f"Chargement du modèle depuis : {checkpoint_path}")
# On charge les poids du réseau de neurones et on lui associe le nouvel environnement
model = TD3.load(checkpoint_path, env=env, device="cuda")

# 3. On reconfigure les callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path='./logs/models/',
    name_prefix='tiago_model'
)
dashboard_callback = TiagoDashboardCallback()
callbacks = CallbackList([checkpoint_callback, dashboard_callback])

print("Reprise de l'entraînement avec les compteurs synchronisés...")

# 4. CRUCIAL : reset_num_timesteps=False pour continuer à cumuler les pas
model.learn(total_timesteps=100000, callback=callbacks, reset_num_timesteps=False)

# Sauvegarde finale
model.save("tiago_td3_model_final")

env.close()
rclpy.shutdown()