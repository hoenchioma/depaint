# DePAint: A Decentralized Safe Multi-Agent Reinforcement Learning Algorithm considering Peak and Average Constraints
### Abstract
The field of safe multi-agent reinforcement learning, despite its potential applications in various domains such as drone delivery and vehicle automation, remains relatively unexplored. Training agents to learn optimal policies that maximize rewards while considering specific constraints can be challenging, particularly in scenarios where having a central controller to coordinate the agents during the training process is not feasible. In this paper, we address the problem of multi-agent policy optimization in a decentralized setting, where agents communicate with their neighbors to maximize the sum of their cumulative rewards while also satisfying each agent's safety constraints. We consider both peak and average constraints. In this scenario, there is no central controller coordinating the agents and both the rewards and constraints are only known to each agent locally/privately. We formulate the problem as a decentralized constrained multi-agent Markov Decision Problem and propose a momentum-based decentralized policy gradient method, DePAint, to solve it. To the best of our knowledge, this is the first privacy-preserving fully decentralized multi-agent reinforcement learning algorithm that considers both peak and average constraints. We also provide theoretical analysis and empirical evaluation of our algorithm in various scenarios and compare its performance to centralized algorithms that consider similar constraints.
### How to run
```bash
python3 -m pip install -r requirements.txt
python3 main.py
```
### Citation
If you find our work useful, please consider citing us!
```bibtex
@article{Hassan2023DePAint,
  author = {Raheeb Hassan and K.M. Shadman Wadith and Md. Mamun or Rashid and Md. Mosaddek Khan},
  title = {DePAint: A Decentralized Safe Multi-Agent Reinforcement Learning Algorithm considering Peak and Average Constraints},
  journal = {arXiv preprint},
  year = {2023},
  eprint = {2310.14348},
  archivePrefix = {arXiv},
  primaryClass = {cs.MA},
  doi = {10.48550/arXiv.2310.14348}
}
```
