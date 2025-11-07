#!/usr/bin/env python3
"""
file: ppo_train.py
author: Tawn Kramer
date: 13 October 2018
notes: ppo2 test from stable-baselines here:
https://github.com/hill-a/stable-baselines
"""

import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import uuid
import tqdm

import gym
import gym_donkeycar
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from utils.sim_starter import start_sim
from algorithms.rl_utils import process_observation_image, extract_action_value, extract_reward_value

VERBOSITY_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50
}

def vb(level_str, args):
    """ return true if the verbosity level is at least level_str """
    return args.verbosity and VERBOSITY_LEVELS[args.verbosity] <= VERBOSITY_LEVELS[level_str]

class VisualizationCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        pygame.init()
        self.font = pygame.font.SysFont(None, 24)
        self.initialized = False
        self.reward_history = []  # Track rewards over time
        self.max_history = 200  # Number of points to display in the plot

    def _on_step(self) -> bool:
        obs = self.locals['new_obs']
        # Process observation using shared utility
        obs_img = process_observation_image(obs)
        
        action = self.locals['actions']
        clipped_action = self.locals['clipped_actions']
        rewards = self.locals['rewards']

        # Extract first element from batch if needed using shared utility
        action = extract_action_value(action)
        clipped_action = extract_action_value(clipped_action)

        # obs_img should now be H x W x C
        w, h = obs_img.shape[:2]
        
        # Scale up the image for better visibility (4x scale to get 640x480 from 160x120)
        scale_factor = 4
        scaled_w, scaled_h = w * scale_factor, h * scale_factor
        
        if not self.initialized:
            self.screen = pygame.display.set_mode((scaled_w, scaled_h + 150))
            self.initialized = True

        # Display observation image
        # Create a surface from the processed image and scale it up
        surface = pygame.surfarray.make_surface(obs_img)
        scaled_surface = pygame.transform.scale(surface, (scaled_w, scaled_h))
        self.screen.blit(scaled_surface, (0, 0))

        # Clear the UI area (bottom portion) before redrawing
        pygame.draw.rect(self.screen, (0, 0, 0), (0, scaled_h, scaled_w, 150))

        # Progress bars
        bar_y = scaled_h + 10
        bar_width = 150
        bar_height = 20
        label_x = 10  # Position labels to the left of bars
        bar_x = 180   # Position bars to the right of labels
        value_x = bar_x + bar_width + 10  # Position values to the right of bars

        # Steer action bar (red) - center at 50% with -1 to 1 range
        steer_value_raw = float(action[0])
        steer_percent = int((steer_value_raw + 1) / 2 * 100)
        steer_label = self.font.render("Steer:", True, (255, 255, 255))
        self.screen.blit(steer_label, (label_x, bar_y))
        pygame.draw.rect(self.screen, (255, 0, 0), (bar_x, bar_y, bar_width, bar_height), 2)
        # Draw from center outward
        center_x = bar_x + bar_width // 2
        if steer_value_raw >= 0:
            # Right side (positive steering)
            width = int((steer_value_raw) * (bar_width // 2))
            pygame.draw.rect(self.screen, (255, 0, 0), (center_x, bar_y, width, bar_height))
        else:
            # Left side (negative steering)
            width = int((abs(steer_value_raw)) * (bar_width // 2))
            pygame.draw.rect(self.screen, (255, 0, 0), (center_x - width, bar_y, width, bar_height))
        steer_text = self.font.render(f"{steer_value_raw:.2f}", True, (255, 255, 255))
        self.screen.blit(steer_text, (value_x, bar_y))

        # Throttle action bar (green) - ranges from 0 to 1, center at 50%
        throttle_value_raw = float(action[1])
        throttle_percent = int(throttle_value_raw * 100)
        throttle_label = self.font.render("Throttle:", True, (255, 255, 255))
        self.screen.blit(throttle_label, (label_x, bar_y + 30))
        pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, bar_y + 30, bar_width, bar_height), 2)
        width = int(throttle_value_raw * bar_width)
        pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, bar_y + 30, width, bar_height))
        throttle_text = self.font.render(f"{throttle_value_raw:.2f}", True, (255, 255, 255))
        self.screen.blit(throttle_text, (value_x, bar_y + 30))

        # Clipped actions
        clipped_y = bar_y + 60

        # Steer clipped bar - center at 50% with -1 to 1 range
        steer_clipped_raw = float(clipped_action[0])
        steer_clipped_percent = int((steer_clipped_raw + 1) / 2 * 100)
        steer_clipped_label = self.font.render("Steer (c):", True, (255, 255, 255))
        self.screen.blit(steer_clipped_label, (label_x, clipped_y))
        pygame.draw.rect(self.screen, (255, 0, 0), (bar_x, clipped_y, bar_width, bar_height), 2)
        # Draw from center outward
        center_x = bar_x + bar_width // 2
        if steer_clipped_raw >= 0:
            # Right side (positive steering)
            width = int((steer_clipped_raw) * (bar_width // 2))
            pygame.draw.rect(self.screen, (255, 0, 0), (center_x, clipped_y, width, bar_height))
        else:
            # Left side (negative steering)
            width = int((abs(steer_clipped_raw)) * (bar_width // 2))
            pygame.draw.rect(self.screen, (255, 0, 0), (center_x - width, clipped_y, width, bar_height))
        steer_clipped_text = self.font.render(f"{steer_clipped_raw:.2f}", True, (255, 255, 255))
        self.screen.blit(steer_clipped_text, (value_x, clipped_y))

        # Throttle clipped bar - ranges from 0 to 1
        throttle_clipped_raw = float(clipped_action[1])
        throttle_clipped_percent = int(throttle_clipped_raw * 100)
        throttle_clipped_label = self.font.render("Throttle (c):", True, (255, 255, 255))
        self.screen.blit(throttle_clipped_label, (label_x, clipped_y + 30))
        pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, clipped_y + 30, bar_width, bar_height), 2)
        width = int(throttle_clipped_raw * bar_width)
        pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, clipped_y + 30, width, bar_height))
        throttle_clipped_text = self.font.render(f"{throttle_clipped_raw:.2f}", True, (255, 255, 255))
        self.screen.blit(throttle_clipped_text, (value_x, clipped_y + 30))

        # Reward text
        reward_val = extract_reward_value(rewards)
        reward_text = self.font.render(f"Reward: {reward_val:.2f}", True, (255, 255, 255))
        self.screen.blit(reward_text, (10, clipped_y + 60))

        # Add reward to history
        self.reward_history.append(reward_val)
        if len(self.reward_history) > self.max_history:
            self.reward_history.pop(0)

        # Draw scrolling reward plot
        screen_height = scaled_h + 150
        self._draw_reward_plot(scaled_w, screen_height)

        pygame.display.flip()

        # Handle events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        return True

    def _draw_reward_plot(self, screen_width, screen_height):
        """Draw a scrolling reward plot on the bottom right of the screen."""
        if len(self.reward_history) < 2:
            return

        import numpy as np

        # Plot dimensions
        plot_width = 240
        plot_height = 120
        plot_x = screen_width - plot_width - 10
        plot_y = screen_height - plot_height - 10

        # Draw plot background
        pygame.draw.rect(self.screen, (40, 40, 40), (plot_x, plot_y, plot_width, plot_height))
        pygame.draw.rect(self.screen, (100, 100, 100), (plot_x, plot_y, plot_width, plot_height), 2)

        # Draw grid lines
        for i in range(5):
            y = plot_y + (plot_height * i) // 4
            pygame.draw.line(self.screen, (60, 60, 60), (plot_x, y), (plot_x + plot_width, y), 1)

        # Calculate min and max rewards for scaling
        rewards_array = np.array(self.reward_history)
        min_reward = rewards_array.min()
        max_reward = rewards_array.max()
        reward_range = max_reward - min_reward if max_reward > min_reward else 1

        # Draw reward line
        for i in range(len(self.reward_history) - 1):
            # Normalize rewards to plot height
            y1 = plot_y + plot_height - ((self.reward_history[i] - min_reward) / reward_range * plot_height)
            y2 = plot_y + plot_height - ((self.reward_history[i + 1] - min_reward) / reward_range * plot_height)

            # Distribute points across plot width
            x1 = plot_x + (i / max(len(self.reward_history) - 1, 1)) * plot_width
            x2 = plot_x + ((i + 1) / max(len(self.reward_history) - 1, 1)) * plot_width

            # Draw line in yellow
            pygame.draw.line(self.screen, (255, 255, 0), (x1, y1), (x2, y2), 2)

        # Draw labels
        title = self.font.render("Reward", True, (255, 255, 255))
        self.screen.blit(title, (plot_x + 5, plot_y - 25))

        min_text = self.font.render(f"Min: {min_reward:.2f}", True, (150, 150, 150))
        self.screen.blit(min_text, (plot_x + 5, plot_y + plot_height + 5))

        max_text = self.font.render(f"Max: {max_reward:.2f}", True, (150, 150, 150))
        self.screen.blit(max_text, (plot_x + 5, plot_y + plot_height + 25))

if __name__ == "__main__":
    # Initialize the donkey environment
    # where env_name one of:
    env_list = [
        "donkey-warehouse-v0",
        "donkey-generated-roads-v0",
        "donkey-avc-sparkfun-v0",
        "donkey-generated-track-v0",
        "donkey-roboracingleague-track-v0",
        "donkey-waveshare-v0",
        "donkey-minimonaco-track-v0",
        "donkey-warren-track-v0",
        "donkey-thunderhill-track-v0",
        "donkey-circuit-launch-track-v0",
    ]

    parser = argparse.ArgumentParser(description="ppo_train")
    parser.add_argument(
        "--sim",
        type=str,
        default="sim_path",
        help="path to unity simulator. maybe be left at manual if you would like to start the sim on your own.",
    )
    parser.add_argument("--port", type=int, default=9091, help="port to use for tcp")
    parser.add_argument("--test", action="store_true", help="load the trained model and play")
    parser.add_argument("--multi", action="store_true", help="start multiple sims at once")
    parser.add_argument(
        "--env_name", type=str, default="donkey-circuit-launch-track-v0", help="name of donkey sim environment", choices=env_list
    )
    parser.add_argument("--verbosity", type=str, default="WARNING", choices=list(VERBOSITY_LEVELS.keys()), help="set the logging verbosity level")

    parser.add_argument("--visualize", action="store_true", help="enable visualization callback during training")

    args = parser.parse_args()

    if args.sim == "sim_path" and args.multi:
        print("you must supply the sim path with --sim when running multiple environments")
        exit(1)

    env_id = args.env_name

    conf = {
        #"exe_path": args.sim,
        "host": "127.0.0.1",
        "port": args.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "me",
        "font_size": 100,
        "racer_name": "PPO",
        "country": "USA",
        "bio": "Learning to drive w PPO RL",
        "guid": str(uuid.uuid4()),
        "max_cte": 10,
    }

    callback_list = []
    if args.visualize:
        callback_list += [VisualizationCallback()]

    if args.test:
        # Make an environment test our trained policy
        env = start_sim(args.env_name, port=args.port, conf=conf)

        model = PPO.load("ppo_donkey")

        obs = env.reset()
        pbar = tqdm.tqdm(range(1000))
        for i in pbar:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if vb("DEBUG", args):
                pbar.set_description(f"v={action[0]:.2f}, θ={action[1]:.2f}, R={reward:.2f}")
            env.render()
            if done:
                obs = env.reset()

        print("done testing")

    else:
        # make gym env
        env = start_sim(args.env_name, port=args.port, conf=conf)

        # create cnn policy
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./output/tensorboard/")

        # set up model in learning mode with goal number of timesteps to complete
        model.learn(total_timesteps=10000, callback=callback_list)

        obs = env.reset()

        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)

            try:
                env.render()
            except Exception as e:
                print(e)
                print("failure in render, continuing...")

            if done:
                obs = env.reset()

            if i % 100 == 0:
                print("saving...")
                model.save("ppo_donkey")

        # Save the agent
        model.save("ppo_donkey")
        print("done training")

    env.close()
