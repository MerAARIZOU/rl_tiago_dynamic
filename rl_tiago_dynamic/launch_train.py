import rclpy
from tiago_env import TiagoEnv
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor

# 1. Création d'un Callback personnalisé pour un affichage clair des Échecs/Succès
class TiagoDashboardCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0

    def _on_step(self) -> bool:
        # On détecte la fin d'un épisode
        if self.locals.get("dones")[0]:
            self.episode_count += 1
            # On récupère la dernière récompense reçue pour analyser la fin de l'épisode
            last_reward = self.locals.get("rewards")[0]
            
            print("\n" + "="*50)
            if last_reward > 100.0:  # Jackpot (+200)
                self.success_count += 1
                print(f"[ÉPISODE {self.episode_count}] REUSSI ! Cible atteinte.")
            elif last_reward < -50.0:  # Collision (-100)
                self.collision_count += 1
                print(f"[ÉPISODE {self.episode_count}] CRASH ! Le robot a percuté un obstacle.")
            else:
                print(f"[ÉPISODE {self.episode_count}] TIMEOUT ! Fin du temps imparti.")
            
            # Calcul du taux de réussite global
            success_rate = (self.success_count / self.episode_count) * 100
            print(f"   -> Total Épisodes : {self.episode_count}")
            print(f"   -> Succès : {self.success_count} | Crashs : {self.collision_count}")
            print(f"   -> Taux de réussite actuel : {success_rate:.1f}%")
            print("="*50 + "\n")
            
        return True

# 2. Initialisation du nœud ROS 2 et de l'environnement
rclpy.init()

# On enveloppe l'environnement avec Monitor pour collecter les statistiques internes
env = Monitor(TiagoEnv(), filename="./logs/monitor.csv")

# 3. Initialisation du modèle TD3 (verbose=1 pour activer les logs automatiques de SB3)
model = TD3("MlpPolicy", env, device="cuda", verbose=1, learning_rate=0.001)

# 4. Configuration des Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path='./logs/models/',
    name_prefix='tiago_model'
)
dashboard_callback = TiagoDashboardCallback()

# On fusionne nos deux callbacks
callbacks = CallbackList([checkpoint_callback, dashboard_callback])

print("Début de l'entraînement. Regardez les logs ci-dessous...")
# Lancement de l'apprentissage (100 000 steps)
model.learn(total_timesteps=100000, callback=callbacks)

# Sauvegarde finale
model.save("tiago_td3_model")
print("Entraînement terminé et modèle final sauvegardé !")