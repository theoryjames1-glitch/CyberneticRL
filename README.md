# 🧭 Cybernetic Reinforcement Learning (CyberneticRL)

This repository collects **theories, formalisms, and code** for **Cybernetic Reinforcement Learning** —  
a framework where agents regulate themselves using **internal cybernetic signals** in addition to external rewards.  

Traditional RL maximizes expected return from the environment.  
**CyberneticRL generalizes this** by introducing a *cybernetic reward factor* that reflects the agent’s own viability (stability, plasticity, diversity, attractor robustness).  

---

## 📖 Core Idea

Instead of relying only on environment reward:

\[
J(\pi) = \mathbb{E}\Big[\sum_t \gamma^t r_t^{\text{env}}\Big]
\]

CyberneticRL uses a **cybernetic reward multiplier**:

\[
r_t^{\text{total}} = r_t^{\text{env}} \cdot R_t^{\text{cyber}}
\]

where \(R_t^{\text{cyber}}\) is derived from internal signals (grad norms, entropy, contraction, etc.).

---

## 🔑 Cybernetic Signals

- **R_task** → Task reward (external environment or metric).  
- **R_stab** → Stability (bounded gradients, spectral norms).  
- **R_div** → Diversity (attention entropy, behavioral variety).  
- **R_plast** → Plasticity (healthy adaptation, non-frozen learning).  
- **R_attr** → Attractor fit (robust hidden state convergence).  
- *(Optional)* **R_bci** → Brain-computer or human feedback.  

These signals are combined via **weighted sum**, **fuzzy logic**, or **adaptive weighting**.

---

## ⚙️ Adaptation Law

Parameters evolve as:

\[
\theta_{t+1} = \theta_t + \alpha \,\nabla_\theta \big( r_t^{\text{env}} \cdot R_t^{\text{cyber}} \big)
\]

The cybernetic reward acts as a **homeostatic regulator**:  
- Suppressing updates when unstable.  
- Amplifying updates when task-aligned & healthy.  

---

## 🌀 Dual Feedback Loops

- **Outer loop (task)** → Agent interacts with environment → gets \(r_t^{\text{env}}\).  
- **Inner loop (cybernetic)** → Agent monitors itself → computes \(R_t^{\text{cyber}}\).  

This forms a **cybernetic control system** for learning.

---

## 🧮 Example Code Snippets

### Cybernetic Reward Computer

```python
reward_comp = CyberneticRewardComputer()

signals = {
    "R_task": np.exp(-loss.item()),
    "R_stab": stability_monitor(model),
    "R_div": diversity_monitor(outputs),
    "R_plast": plasticity_monitor(model),
    "R_attr": attractor_monitor(outputs)
}

cyber_reward = reward_comp.compute(signals)[0]
cyber_loss = loss * cyber_reward
````

---

### CyberneticRL Loop (pseudo-code)

```python
for episode in range(num_episodes):
    state = env.reset()
    for t in range(max_steps):
        action = policy(state)
        next_state, reward_env, done, _ = env.step(action)

        # internal cybernetic signals
        signals = monitor.compute(model, loss, inputs, outputs)

        # combine with environment reward
        cyber_reward = reward_comp.compute(signals)[0]
        total_reward = reward_env * cyber_reward

        # RL update
        update_policy(total_reward, state, action, next_state)

        state = next_state
        if done:
            break
```

---

## 🔬 Experiments

* **Compare** CyberneticRL vs standard RL.
* Track signals (`R_stab`, `R_div`, etc.) over time.
* Ablations: run with subsets of signals.
* Plot phase transitions (when cyber signals cross thresholds).

---

## 📂 Repo Structure

```
CyberneticRL-Theory/
│
├── docs/                  # extended theory pages
│   ├── theory_cyberneticrl.md
│   ├── cybernetic_signals.md
│   ├── fuzzy_logic.md
│   ├── bci_integration.md
│   └── experiments.md
│
├── examples/              # runnable prototypes
│   ├── cybernetic_reward.py
│   ├── cybernetic_monitor.py
│   └── cybernetic_rl_loop.py
│
└── README.md              # this file
```

---

## 📌 Roadmap

* [x] Core theory of CyberneticRL
* [x] Signal monitors (stability, diversity, plasticity, attractors)
* [x] Cybernetic reward computer
* [ ] Fuzzy logic integration
* [ ] BCI feedback integration
* [ ] Full RL benchmark experiments

---

## 📚 References

* Cybernetics: W. Ross Ashby, *Design for a Brain* (1952).
* Control Theory & RL: Sutton & Barto, *Reinforcement Learning* (1998).
* Recent works on stability in transformers, entropy regularization, RLHF.

---

✨ *CyberneticRL = RL with self-awareness. The agent doesn’t just chase rewards — it regulates itself like a living system.*
