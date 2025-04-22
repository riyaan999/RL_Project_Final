# Exploring Reinforcement Learning as an Alternative for Diabetic Retinopathy Classification

This research project explores the application of **Reinforcement Learning (RL)** — specifically, **Proximal Policy Optimization (PPO)** — for classifying **Diabetic Retinopathy (DR)** severity levels from retinal fundus images. The goal is to evaluate PPO as an alternative to traditional supervised learning approaches like CNNs for vision-based medical diagnosis.

---

## 📂 Dataset

- **Source:** A Gaussian-filtered version of the APTOS 2019 Blindness Detection dataset
- **Classes:**  
  - `0 - No_DR`  
  - `1 - Mild`  
  - `2 - Moderate`  
  - `3 - Severe`  
  - `4 - Proliferate_DR`
- **Preprocessing:**  
  - Resized to `3×224×224`  
  - Normalized using ImageNet stats  
  - Augmented with random rotation, horizontal flip, and color jitter

---

## 🧪 Methodology

### Custom Environment
- Built using **Gymnasium** (`DiabeticRetinopathyEnv`)
- Each episode samples 20 images randomly
- Reward shaping includes:
  - +8 for correct classification
  - −3 base penalty + distance-based penalty
  - +3 consistency bonus
  - −0.5 confidence penalty
  - Progress-based reward scaling

### PPO Setup
- Framework: `stable-baselines3`
- Policy: `CnnPolicy` with custom CNN extractor
- Timesteps: `100,000`
- Clip Range: `0.2`, Entropy Coeff: `0.02`
- Learning Rate: `5e-5`, Batch Size: `64`, n_steps: `512`
- Epochs: `10`
- TensorBoard for logging and monitoring

---

## 🔄 Training Pipeline

1. Load dataset using `ImageFolder` from `torchvision`
2. Wrap environment with `DummyVecEnv`, `VecNormalize`, `VecMonitor`
3. Train PPO agent with custom CNN features
4. Periodic evaluation using `EvalCallback`
5. Visualize performance using metrics and plots

---

## 📈 Results

- **Mean Eval Reward:** `1.84 ± 1.40`
- **Overall Accuracy:** `46%`
- **Macro F1-Score:** `0.16`
- **Explained Variance (Value Function):** `-0.142`
- Strong classification bias toward `No_DR`
- Underperformance on minority classes like `Severe` and `Proliferate_DR`

---

## 📌 Conclusion

PPO showed potential in learning classification behavior from shaped rewards, but struggled with class imbalance and value estimation stability. While it underperformed compared to a CNN baseline, it demonstrated how reinforcement learning can be adapted to medical image analysis tasks. Future work should focus on:
- Reward rebalancing strategies
- Class-weighted exploration or sampling
- Hybrid CNN + PPO architectures

---

## 🛠️ Requirements

- Python 3.9+
- PyTorch
- Gymnasium
- Stable-Baselines3
- Torchvision
- TensorBoard (optional)
- Matplotlib / Seaborn (for evaluation)

---

## 👤 Author

**Riyaan Chatterjee**  

---
