import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import psutil
import gc
import random

class DiabeticRetinopathyEnv(gym.Env):
    """Custom Environment for Diabetic Retinopathy Classification with enhanced features"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path, batch_size=32, memory_limit_gb=4, episode_length=20):
        super(DiabeticRetinopathyEnv, self).__init__()
        
        self.batch_size = batch_size
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        self.episode_length = episode_length  # Set short episodes for PPO
        
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(3, 224, 224),
            dtype=np.float32
        )

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.data_path = data_path
        self.dataset = datasets.ImageFolder(
            root=os.path.join(data_path, 'gaussian_filtered_images/gaussian_filtered_images'),
            transform=self.transform
        )

        self.indices = list(range(len(self.dataset)))
        self.episode_data = []
        self.current_idx = 0
        self.total_samples = len(self.dataset)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.action_history = []
        self.last_correct = False
        
        self.check_memory_usage()

        # Sample a new episode randomly
        self.episode_data = random.sample(self.indices, self.episode_length)
        self.current_idx = 0

        try:
            image, _ = self.dataset[self.episode_data[self.current_idx]]
            return image.numpy(), {}
        except Exception as e:
            print(f"Error in reset: {str(e)}")
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        if self.current_idx >= len(self.episode_data):
            return None, 0.0, True, False, {}

        image, true_label = self.dataset[self.episode_data[self.current_idx]]
        self.current_idx += 1
        image_np = image.numpy()

        try:
            confidence_penalty = 0.0
            if hasattr(self, 'action_history'):
                self.action_history.append(action)
                if len(self.action_history) > 5:
                    self.action_history.pop(0)
                    changes = sum(1 for i in range(len(self.action_history)-1)
                                  if self.action_history[i] != self.action_history[i+1])
                    confidence_penalty = changes * 0.5

            # Base reward and distance penalty
            base_reward = 8.0 if action == true_label else -3.0
            if action != true_label:
                distance_penalty = abs(action - true_label) * 1.5
                base_reward -= distance_penalty

            # Bonus for consistency in correct actions
            if self.last_correct and action == true_label:
                base_reward += 3.0
            self.last_correct = (action == true_label)

            # Progress-based difficulty (for curriculum)
            progress = self.current_idx / self.episode_length
            difficulty_factor = 0.5 + 0.5 * (1 - np.exp(-3 * progress))

            # Final shaped reward
            reward = (base_reward - confidence_penalty) * difficulty_factor

        except Exception as e:
            print(f"Error in reward calculation: {str(e)}")
            reward = 0.0

        terminated = self.current_idx >= self.episode_length
        truncated = False

        return image_np, reward, terminated, truncated, {'true_label': true_label}

    def render(self, mode='human'):
        pass

    def close(self):
        try:
            if hasattr(self, 'dataset'):
                del self.dataset
            if hasattr(self, 'action_history'):
                del self.action_history
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error in close: {str(e)}")

    def check_memory_usage(self):
        try:
            process = psutil.Process()
            memory_usage = process.memory_info().rss
            if memory_usage > self.memory_limit_bytes:
                print("Warning: Approaching memory limit, clearing cache...")
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            print(f"Error in memory check: {str(e)}")