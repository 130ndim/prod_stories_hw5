import os
import sys
import shutil
from gym import spaces

import numpy as np

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray import tune
from PIL import Image

sys.path.append(os.path.abspath("mapgen"))
os.environ["PYTHONPATH"] = os.path.abspath("mapgen")

from mapgen.map import Move
from mapgen import Dungeon

import wandb


class ModifiedDungeon(Dungeon):
    """Use this class to change the behavior of the original env
    (e.g. remove the trajectory from observation, like here)"""

    def __init__(
            self,
            width=64,
            height=64,
            max_rooms=25,
            min_room_xy=10,
            max_room_xy=25,
            observation_size: int = 11,
            vision_radius: int = 5,
            max_steps: int = 2000
    ):
        super().__init__(
            width=width,
            height=height,
            max_rooms=max_rooms,
            min_room_xy=min_room_xy,
            max_room_xy=max_room_xy,
            observation_size=observation_size,
            vision_radius=vision_radius,
            max_steps=max_steps
        )

    def step(self, action: int):
        action = Move(action - 1)
        observation, explored, done, moved, is_new = self._map.step(self._agent, action,
                                                                    self.observation_size)

        moved_r = moved * 2 - 1
        is_new_r = is_new * 2 - 1
        if moved and not is_new:
            moved_r = -1

        reward = (10 * explored + moved_r + is_new_r - 1) / self._map._visible_cells

        info = {
            "step": self._step,
            "total_cells": self._map._visible_cells,
            "total_explored": self._map._total_explored,
            "new_explored": explored,
            "avg_explored_per_step": self._map._total_explored / self._step,
            "moved": moved,
            "is_new": is_new
        }
        self._step += 1

        return observation, reward, done or self._step == self._max_steps, info
    

if __name__ == "__main__":

    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    tune.register_env("ModifiedDungeon", lambda config: ModifiedDungeon(**config))

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 1
    config["log_level"] = "INFO"
    config["framework"] = "torch"
    config["env"] = "ModifiedDungeon"
    config["env_config"] = {
        "width": 20,
        "height": 20,
        "max_rooms": 3,
        "min_room_xy": 5,
        "max_room_xy": 10,
        "observation_size": 11,
        "vision_radius": 5,
        "max_steps": 400
    }

    config["model"] = {
        "conv_filters": [
            [16, (3, 3), 2],
            [32, (3, 3), 2],
            [32, (3, 3), 1],
        ],
        "post_fcnet_hiddens": [32, 32],
        "post_fcnet_activation": "relu",
        "vf_share_layers": False,
    }

    config["rollout_fragment_length"] = 100
    config["entropy_coeff"] = 0.1
    config["lambda"] = 0.95
    config["vf_loss_coeff"] = 1.0
    config['num_gpus'] = 0
    config['num_gpus_per_worker'] = 0

    agent = ppo.PPOTrainer(config)

    N_ITER = 500
    s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

    wb = wandb.init(project='vac', config=config)
    CHECKPOINT_ROOT = os.path.join(wb.dir, "ckpts")
    shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

    ray_results = os.getenv("HOME") + "/ray_results1/"
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)
    for n in range(N_ITER):
        result = agent.train()
        file_name = agent.save(CHECKPOINT_ROOT)

        print(s.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            file_name
        ))
        wandb.log({
            "episode_reward_min": result["episode_reward_min"],
            "episode_reward_mean": result["episode_reward_mean"],
            "episode_reward_max": result["episode_reward_max"],
            "episode_len_mean": result["episode_len_mean"],
            "entropy": result["info"]["learner"]["default_policy"]["learner_stats"]["entropy"]
        })
        # print(result)

        # sample trajectory
        if (n + 1) % 5 == 0:
            env = ModifiedDungeon(20, 20, 3, min_room_xy=5, max_room_xy=10,
                                  max_steps=400)
            obs = env.reset()
            # Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST).save('tmp.png')

            frames = []

            for _ in range(500):
                action = agent.compute_single_action(obs)

                frame = Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST)
                frames.append(np.array(frame))

                obs, reward, done, info = env.step(action)
                if done:
                    break
            vid = np.array(frames).transpose((0, 3, 1, 2))
            wandb.log({
                "gif": wandb.Video(vid, fps=30)
            })
            # frames[0].save(f"out.gif", save_all=True, append_images=frames[1:], loop=0, duration=1000/60)