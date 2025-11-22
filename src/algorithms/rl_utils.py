#!/usr/bin/env python3
"""
Shared utilities for RL algorithms
Contains common components for PPO, SAC, and other algorithms
"""

import os
import random
from datetime import datetime
from typing import Tuple, Optional

import numpy as np
import pygame
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize network layer with orthogonal initialization"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def set_random_seeds(seed: int, torch_deterministic: bool = True):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def setup_logging_dirs(log_dir: str, model_dir: str) -> Tuple[str, str]:
    """Create logging directories with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"{log_dir}/{timestamp}"
    model_path = f"{model_dir}/{timestamp}"
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    return log_path, model_path


class CNNFeatureExtractor(nn.Module):
    """
    Shared CNN feature extractor for visual observations
    Used by both PPO and SAC for DonkeyEnv's camera input
    """
    def __init__(self, observation_space):
        super().__init__()
        
        # Determine input shape
        # DonkeyEnv provides HxWxC images
        obs_shape = observation_space.shape
        if len(obs_shape) == 3:
            # Assume HxWxC format, convert to CxHxW for PyTorch
            self.input_channels = obs_shape[2]
            self.height = obs_shape[0]
            self.width = obs_shape[1]
        else:
            raise ValueError(f"Unexpected observation shape: {obs_shape}")
        
        # CNN layers
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.height, self.width)
            self.feature_dim = self.cnn(dummy_input).shape[1]
    
    def forward(self, x):
        """
        Extract features from observations
        Args:
            x: Observations in HxWxC or BxHxWxC format
        Returns:
            Features: Flattened feature vector(s)
        """
        # Convert HxWxC to CxHxW and normalize to [0, 1]
        if len(x.shape) == 4:  # Batch of images
            x = x.permute(0, 3, 1, 2)  # BxHxWxC -> BxCxHxW
        else:
            x = x.permute(2, 0, 1)  # HxWxC -> CxHxW
        x = x.float() / 255.0
        return self.cnn(x)


def prepare_observation(obs, device):
    """Convert observation to tensor on device"""
    if not isinstance(obs, np.ndarray):
        obs = np.array(obs)
    return torch.from_numpy(obs).float().to(device)


def clip_action_to_space(action, action_space):
    """Clip actions to action space bounds"""
    action_np = action.cpu().numpy() if torch.is_tensor(action) else action
    
    # Ensure dtype matches action space
    if action_np.dtype != action_space.dtype:
        action_np = action_np.astype(action_space.dtype)
    
    # Clip to bounds
    action_np = np.clip(
        action_np,
        action_space.low,
        action_space.high
    )
    
    return action_np


def extract_lap_time_metrics(info, idx):
    """
    Extract lap time metrics from info dict (can be called even when episode not done)
    Returns dict with lap time metrics (or None if no new lap time)
    """
    metrics = {}
    
    # Handle list of dicts (PufferLib)
    if isinstance(info, list):
        if idx < len(info):
            env_info = info[idx]
            if "last_lap_time" in env_info and env_info["last_lap_time"] > 0.0:
                metrics["lap_time"] = env_info["last_lap_time"]
            if "lap_count" in env_info:
                metrics["lap_count"] = env_info["lap_count"]
        return metrics if metrics else None
    
    if isinstance(info, dict):
        # Lap metrics - check for new lap time
        if "last_lap_time" in info:
            lap_time = info["last_lap_time"][idx] if hasattr(info["last_lap_time"], "__getitem__") else info["last_lap_time"]
            if lap_time > 0.0:
                metrics["lap_time"] = lap_time
        
        if "lap_count" in info:
            lap_count = info["lap_count"][idx] if hasattr(info["lap_count"], "__getitem__") else info["lap_count"]
            metrics["lap_count"] = lap_count
    
    return metrics if metrics else None


def extract_episode_metrics(info, idx, done):
    """
    Extract episode metrics from info dict
    Returns dict with episode metrics (or None if episode not done)
    """
    metrics = {}
    
    if not done:
        return None
    
    # Handle list of dicts (PufferLib)
    if isinstance(info, list):
        if idx < len(info):
            env_info = info[idx]
            
            if "episode" in env_info:
                metrics["reward"] = env_info["episode"]["r"]
                metrics["length"] = env_info["episode"]["l"]
            
            # Cross track error
            if "cte" in env_info:
                metrics["cte"] = abs(env_info["cte"])
            
            # Speed metrics
            if "speed" in env_info:
                metrics["speed"] = env_info["speed"]
            
            if "forward_vel" in env_info:
                metrics["forward_vel"] = env_info["forward_vel"]
            
            # Collision detection
            if "hit" in env_info:
                metrics["hit"] = 1.0 if env_info["hit"] != "none" else 0.0
            
            # Lap metrics
            if "last_lap_time" in env_info and env_info["last_lap_time"] > 0.0:
                metrics["lap_time"] = env_info["last_lap_time"]
            
            if "lap_count" in env_info:
                metrics["lap_count"] = env_info["lap_count"]
                
        return metrics if metrics else None
    
    if "episode" in info:
        ep_info = info["episode"][idx]
        metrics["reward"] = ep_info["r"]
        metrics["length"] = ep_info["l"]
    
    if isinstance(info, dict):
        # Cross track error
        if "cte" in info:
            cte_val = info["cte"][idx] if hasattr(info["cte"], "__getitem__") else info["cte"]
            metrics["cte"] = abs(cte_val)
        
        # Speed metrics
        if "speed" in info:
            speed_val = info["speed"][idx] if hasattr(info["speed"], "__getitem__") else info["speed"]
            metrics["speed"] = speed_val
        
        if "forward_vel" in info:
            fwd_vel = info["forward_vel"][idx] if hasattr(info["forward_vel"], "__getitem__") else info["forward_vel"]
            metrics["forward_vel"] = fwd_vel
        
        # Collision detection
        if "hit" in info:
            hit_val = info["hit"][idx] if hasattr(info["hit"], "__getitem__") else info["hit"]
            metrics["hit"] = 1.0 if hit_val != "none" else 0.0
        
        # Lap metrics
        if "last_lap_time" in info:
            lap_time = info["last_lap_time"][idx] if hasattr(info["last_lap_time"], "__getitem__") else info["last_lap_time"]
            if lap_time > 0.0:
                metrics["lap_time"] = lap_time
        
        if "lap_count" in info:
            lap_count = info["lap_count"][idx] if hasattr(info["lap_count"], "__getitem__") else info["lap_count"]
            metrics["lap_count"] = lap_count
    
    return metrics if metrics else None


class EpisodeMetricsLogger:
    """Helper class to accumulate and log episode metrics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_cte = []
        self.episode_speed = []
        self.episode_forward_vel = []
        self.episode_hits = []
        self.episode_lap_times = []
        self.episode_lap_counts = []
    
    def add_metrics(self, metrics):
        """Add metrics from a single episode"""
        if metrics is None:
            return
        
        if "reward" in metrics:
            self.episode_rewards.append(metrics["reward"])
        if "length" in metrics:
            self.episode_lengths.append(metrics["length"])
        if "cte" in metrics:
            self.episode_cte.append(metrics["cte"])
        if "speed" in metrics:
            self.episode_speed.append(metrics["speed"])
        if "forward_vel" in metrics:
            self.episode_forward_vel.append(metrics["forward_vel"])
        if "hit" in metrics:
            self.episode_hits.append(metrics["hit"])
        if "lap_time" in metrics:
            self.episode_lap_times.append(metrics["lap_time"])
        if "lap_count" in metrics:
            self.episode_lap_counts.append(metrics["lap_count"])
    
    def get_metrics_dict(self):
        """Get all metrics as dictionary"""
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_cte": self.episode_cte,
            "episode_speed": self.episode_speed,
            "episode_forward_vel": self.episode_forward_vel,
            "episode_hits": self.episode_hits,
            "episode_lap_times": self.episode_lap_times,
            "episode_lap_counts": self.episode_lap_counts,
        }
    
    def log_to_tensorboard(self, writer, global_step):
        """Log all accumulated metrics to TensorBoard"""
        # Episode Performance
        if len(self.episode_rewards) > 0:
            rewards_arr = np.array(self.episode_rewards)
            lengths_arr = np.array(self.episode_lengths)
            writer.add_scalar("charts/episodic_return", np.mean(rewards_arr), global_step)
            writer.add_scalar("charts/episodic_length", np.mean(lengths_arr), global_step)
        
        # Driving Performance
        if len(self.episode_cte) > 0:
            cte_arr = np.array(self.episode_cte)
            writer.add_scalar("driving/cross_track_error", np.mean(cte_arr), global_step)
            writer.add_scalar("driving/cte_std", np.std(cte_arr), global_step)
        
        if len(self.episode_speed) > 0:
            speed_arr = np.array(self.episode_speed)
            writer.add_scalar("driving/speed", np.mean(speed_arr), global_step)
        
        if len(self.episode_forward_vel) > 0:
            fwd_vel_arr = np.array(self.episode_forward_vel)
            writer.add_scalar("driving/forward_velocity", np.mean(fwd_vel_arr), global_step)
        
        if len(self.episode_hits) > 0:
            hits_arr = np.array(self.episode_hits)
            writer.add_scalar("driving/collision_rate", np.mean(hits_arr), global_step)
        
        # Lap Performance
        if len(self.episode_lap_times) > 0:
            lap_times_arr = np.array(self.episode_lap_times)
            writer.add_scalar("laps/lap_time_mean", np.mean(lap_times_arr), global_step)
            writer.add_scalar("laps/lap_time_min", np.min(lap_times_arr), global_step)
            writer.add_scalar("laps/lap_time_std", np.std(lap_times_arr), global_step)
        
        if len(self.episode_lap_counts) > 0:
            lap_counts_arr = np.array(self.episode_lap_counts)
            writer.add_scalar("laps/completed_laps", np.sum(lap_counts_arr), global_step)
    
    def print_summary(self):
        """Print summary of metrics"""
        lines = []
        
        if len(self.episode_rewards) > 0:
            rewards_arr = np.array(self.episode_rewards)
            lengths_arr = np.array(self.episode_lengths)
            lines.append(f"  Reward: {np.mean(rewards_arr):.2f} ± {np.std(rewards_arr):.2f}")
            lines.append(f"  Length: {np.mean(lengths_arr):.1f}")
        
        if len(self.episode_cte) > 0:
            cte_arr = np.array(self.episode_cte)
            speed_arr = np.array(self.episode_speed)
            hits_arr = np.array(self.episode_hits)
            lines.append(f"  CTE: {np.mean(cte_arr):.3f} | Speed: {np.mean(speed_arr):.2f} | Collisions: {np.mean(hits_arr):.2%}")
        
        if len(self.episode_lap_times) > 0:
            lap_times_arr = np.array(self.episode_lap_times)
            lines.append(f"  Best Lap: {np.min(lap_times_arr):.2f}s | Avg Lap: {np.mean(lap_times_arr):.2f}s")
        
        return "\n".join(lines)


def process_observation_image(obs):
    """
    Process observation image for visualization
    Converts various formats to HxWxC uint8 numpy array
    """
    # Process observation image
    try:
        arr = np.array(obs)
    except Exception:
        arr = obs
    
    # Handle batch dimension - take first environment
    if hasattr(arr, 'ndim') and arr.ndim == 4:
        arr = arr[0]
    
    # Convert channels-first (C, H, W) to channels-last (H, W, C)
    if hasattr(arr, 'ndim') and arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        try:
            arr = arr.transpose(1, 2, 0)
        except Exception:
            pass
    
    # Handle grayscale 2D image
    if hasattr(arr, 'ndim') and arr.ndim == 2:
        try:
            arr = np.stack([arr, arr, arr], axis=-1)
        except Exception:
            pass
    
    # Ensure uint8 dtype
    if hasattr(arr, 'dtype'):
        try:
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
        except Exception:
            pass
    
    # Rotate 90 degrees clockwise
    try:
        arr = np.rot90(arr, k=1)
    except Exception:
        pass
    
    return arr


def extract_action_value(action):
    """Extract action value from tensor/array, handling batch dimensions"""
    if hasattr(action, 'cpu'):
        action = action.cpu().numpy()
    if hasattr(action, 'ndim') and action.ndim > 1:
        action = action[0]
    return action


def extract_reward_value(reward):
    """Extract reward value from tensor/array"""
    if hasattr(reward, 'item'):
        return reward.item()
    elif hasattr(reward, '__len__') and len(reward) > 0:
        return float(reward[0])
    else:
        return float(reward)


class VisualizationWindow:
    """Pygame visualization window for training"""
    def __init__(self, algorithm_name="RL", port=None):
        pygame.init()
        self.font = pygame.font.SysFont(None, 24)
        self.initialized = False
        self.reward_history = []
        self.max_history = 200
        self.algorithm_name = algorithm_name
        self.port = port
    
    def update(self, obs, action, clipped_action, reward):
        """Update the visualization window with new data"""
        # Process observation
        obs_img = process_observation_image(obs)
        
        # Extract action and reward values
        action = extract_action_value(action)
        clipped_action = extract_action_value(clipped_action)
        reward_val = extract_reward_value(reward)
        
        # Setup window dimensions
        w, h = obs_img.shape[:2]
        scale_factor = 4
        scaled_w, scaled_h = w * scale_factor, h * scale_factor
        
        if not self.initialized:
            self.screen = pygame.display.set_mode((scaled_w, scaled_h + 150))
            # Set window title with algorithm name and port
            title_parts = [self.algorithm_name]
            if self.port is not None:
                title_parts.append(f"Port {self.port}")
            pygame.display.set_caption(" | ".join(title_parts))
            self.initialized = True
        
        # Display observation
        surface = pygame.surfarray.make_surface(obs_img)
        scaled_surface = pygame.transform.scale(surface, (scaled_w, scaled_h))
        self.screen.blit(scaled_surface, (0, 0))
        
        # Clear UI area
        pygame.draw.rect(self.screen, (0, 0, 0), (0, scaled_h, scaled_w, 150))
        
        # Draw UI elements
        bar_y = scaled_h + 10
        bar_width = 150
        bar_height = 20
        label_x = 10
        bar_x = 180
        value_x = bar_x + bar_width + 10
        
        # Steer action bar
        steer_value_raw = float(action[0])
        steer_label = self.font.render("Steer:", True, (255, 255, 255))
        self.screen.blit(steer_label, (label_x, bar_y))
        pygame.draw.rect(self.screen, (255, 0, 0), (bar_x, bar_y, bar_width, bar_height), 2)
        center_x = bar_x + bar_width // 2
        if steer_value_raw >= 0:
            width = int(steer_value_raw * (bar_width // 2))
            pygame.draw.rect(self.screen, (255, 0, 0), (center_x, bar_y, width, bar_height))
        else:
            width = int(abs(steer_value_raw) * (bar_width // 2))
            pygame.draw.rect(self.screen, (255, 0, 0), (center_x - width, bar_y, width, bar_height))
        steer_text = self.font.render(f"{steer_value_raw:.2f}", True, (255, 255, 255))
        self.screen.blit(steer_text, (value_x, bar_y))
        
        # Throttle action bar
        throttle_value_raw = float(action[1])
        throttle_label = self.font.render("Throttle:", True, (255, 255, 255))
        self.screen.blit(throttle_label, (label_x, bar_y + 30))
        pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, bar_y + 30, bar_width, bar_height), 2)
        width = int(throttle_value_raw * bar_width)
        pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, bar_y + 30, width, bar_height))
        throttle_text = self.font.render(f"{throttle_value_raw:.2f}", True, (255, 255, 255))
        self.screen.blit(throttle_text, (value_x, bar_y + 30))
        
        # Clipped actions
        clipped_y = bar_y + 60
        steer_clipped_raw = float(clipped_action[0])
        steer_clipped_label = self.font.render("Steer (c):", True, (255, 255, 255))
        self.screen.blit(steer_clipped_label, (label_x, clipped_y))
        pygame.draw.rect(self.screen, (255, 0, 0), (bar_x, clipped_y, bar_width, bar_height), 2)
        center_x = bar_x + bar_width // 2
        if steer_clipped_raw >= 0:
            width = int(steer_clipped_raw * (bar_width // 2))
            pygame.draw.rect(self.screen, (255, 0, 0), (center_x, clipped_y, width, bar_height))
        else:
            width = int(abs(steer_clipped_raw) * (bar_width // 2))
            pygame.draw.rect(self.screen, (255, 0, 0), (center_x - width, clipped_y, width, bar_height))
        steer_clipped_text = self.font.render(f"{steer_clipped_raw:.2f}", True, (255, 255, 255))
        self.screen.blit(steer_clipped_text, (value_x, clipped_y))
        
        throttle_clipped_raw = float(clipped_action[1])
        throttle_clipped_label = self.font.render("Throttle (c):", True, (255, 255, 255))
        self.screen.blit(throttle_clipped_label, (label_x, clipped_y + 30))
        pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, clipped_y + 30, bar_width, bar_height), 2)
        width = int(throttle_clipped_raw * bar_width)
        pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, clipped_y + 30, width, bar_height))
        throttle_clipped_text = self.font.render(f"{throttle_clipped_raw:.2f}", True, (255, 255, 255))
        self.screen.blit(throttle_clipped_text, (value_x, clipped_y + 30))
        
        # Reward text
        reward_text = self.font.render(f"Reward: {reward_val:.2f}", True, (255, 255, 255))
        self.screen.blit(reward_text, (10, clipped_y + 60))
        
        # Update reward history
        self.reward_history.append(reward_val)
        if len(self.reward_history) > self.max_history:
            self.reward_history.pop(0)
        
        # Draw reward plot
        self._draw_reward_plot(scaled_w, scaled_h + 150)
        
        pygame.display.flip()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        return True
    
    def _draw_reward_plot(self, screen_width, screen_height):
        """Draw a scrolling reward plot"""
        if len(self.reward_history) < 2:
            return
        
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
        
        # Calculate scaling
        rewards_array = np.array(self.reward_history)
        min_reward = rewards_array.min()
        max_reward = rewards_array.max()
        reward_range = max_reward - min_reward if max_reward > min_reward else 1
        
        # Draw reward line
        for i in range(len(self.reward_history) - 1):
            y1 = plot_y + plot_height - ((self.reward_history[i] - min_reward) / reward_range * plot_height)
            y2 = plot_y + plot_height - ((self.reward_history[i + 1] - min_reward) / reward_range * plot_height)
            x1 = plot_x + (i / max(len(self.reward_history) - 1, 1)) * plot_width
            x2 = plot_x + ((i + 1) / max(len(self.reward_history) - 1, 1)) * plot_width
            pygame.draw.line(self.screen, (255, 255, 0), (x1, y1), (x2, y2), 2)
        
        # Draw labels
        title = self.font.render("Reward", True, (255, 255, 255))
        self.screen.blit(title, (plot_x + 5, plot_y - 25))
        min_text = self.font.render(f"Min: {min_reward:.2f}", True, (150, 150, 150))
        self.screen.blit(min_text, (plot_x + 5, plot_y + plot_height + 5))
        max_text = self.font.render(f"Max: {max_reward:.2f}", True, (150, 150, 150))
        self.screen.blit(max_text, (plot_x + 5, plot_y + plot_height + 25))
    
    def close(self):
        """Close the visualization window"""
        if self.initialized:
            pygame.quit()

