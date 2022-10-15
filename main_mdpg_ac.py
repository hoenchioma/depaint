"""
Reference: https://github.com/xylee95/MD-PGT
Paper: "MDPGT: Momentum-based Decentralized Policy Gradient Tracking"
"""
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from rl_utils import *
from tools.tool import *
from momentum_pg_ac import *
from particle_envs import make_particleworld
import torch
import numpy as np
import copy
import os

seed = 0
torch.manual_seed(seed)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def set_args(num_agents=1, beta=0.2, topology='dense'):
    """Set parameters."""
    parser = argparse.ArgumentParser(description='MDPG_AC')
    parser.add_argument('--num_agents', type=int, default=num_agents, help='number of agents')
    parser.add_argument('--num_landmarks', type=int, default=num_agents, help='number of landmarks')
    parser.add_argument('--action_dim', type=int, default=5, help='number of actions')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--critic_lr', type=float, default=1e-3, help='value learning rate')
    parser.add_argument('--grad_lr', type=float, default=3e-4, help='policy learning rate')
    parser.add_argument('--dual_lr', type=float, default=1e-3, help='dual param learning rate')
    parser.add_argument('--constraint', type=float, default=20.0, help='constraint')
    parser.add_argument('--lmbda', type=float, default=0.95, help='lambda for GAE')
    parser.add_argument('--max_eps_len', type=int, default=20, help='number of steps per episode')
    parser.add_argument('--num_episodes', type=int, default=5000, help='number training episodes')
    parser.add_argument('--beta', type=float, default=beta, help='beta for momentum-based variance reduction')
    parser.add_argument('--min_isw', type=float, default=0.0, help='minimum value to set importance weight')
    parser.add_argument('--topology', type=str, default=topology, choices=('dense', 'ring', 'bipartite'))
    parser.add_argument('--init_lr', type=float, default=3e-5, help='policy learning rate for initialization')
    parser.add_argument('--minibatch_size', type=int, default=10, help='number of trajectories for batch gradient')
    parser.add_argument('--init_minibatch_size', type=int, default=10, help='number of trajectories for batch gradient in initialization')
    parser.add_argument('--clip_grad', type=float, default=10, help='gradient clipping')
    parser.add_argument('--dual_clip_grad', type=float, default=100, help='gradient clipping for dual param')
    args = parser.parse_args()
    return args


def run(args, env_name):
    """Run MDPGT algorithm."""
    # timestr = str(time()).replace('.', 'p')
    fpath2 = os.path.join('records', 'mdpgt_logs', str(num_agents) + '_agents', 'beta=' + str(args.beta), topology)
    if not os.path.isdir(fpath2):
        os.makedirs(fpath2)
    else:
        del_file(fpath2)

    writer = SummaryWriter(fpath2)
    sample_envs = []
    sample_env = make_particleworld.make_env(env_name, num_agents=args.num_agents, num_landmarks=args.num_landmarks)
    sample_env.discrete_action_input = True  # set action space to take in discrete numbers 0,1,2,3,4
    sample_env.seed(seed)
    for _ in range(args.num_agents):
        env_copy = copy.deepcopy(sample_env)
        sample_envs.append(env_copy)

    print('observation Space:', sample_env.observation_space)
    print('Action Space:', sample_env.action_space)
    print('Number of agents:', sample_env.n)
    sample_obs = sample_env.reset()
    sample_obs = np.concatenate(sample_obs).ravel()  

    agents = []
    for i in range(args.num_agents):
        agents.append(MomentumPG(num_agents=args.num_agents, state_dim=len(sample_obs), action_dim=args.action_dim, lmbda=args.lmbda,
                                 critic_lr=args.critic_lr, gamma=args.gamma,
                                 device=device, min_isw=args.min_isw, beta=args.beta)) 

    # Load the connectivity weight matrix.
    pi = load_pi(num_agents=args.num_agents, topology=args.topology)

    # Copy old polices.
    old_policies = []
    for agent in agents:
        old_policy = copy.deepcopy(agent.actors)
        old_policies.append(old_policy)

    # Mini-batch initialization.
    prev_v_lists, y_lists = initialization_gt(sample_envs, agents, pi, lr=args.init_lr, minibatch_size=args.init_minibatch_size,
                                                max_eps_len=args.max_eps_len, clip_grad=args.clip_grad)

    # TEST: When topo is dense, complete consensus.
    # numss = []
    # for agent in agents:
    #     nums = []
    #     for idx, actor in enumerate(agent.actors):
    #         a = torch.nn.utils.convert_parameters.parameters_to_vector(actor.parameters())
    #         b = torch.nn.utils.convert_parameters.parameters_to_vector(agents[4].actors[idx].parameters())
    #         num = np.linalg.norm(a.detach().numpy() - b.detach().numpy(), 2)
    #         nums.append(num)
    #     numss.append(nums)
    # print(numss)

    envs = []
    env = make_particleworld.make_env(env_name, num_agents=args.num_agents, num_landmarks=args.num_landmarks)
    env.discrete_action_input = True
    env.seed(seed)
    for _ in range(args.num_agents):
        env_copy = copy.deepcopy(env)
        envs.append(env_copy)

    # return_list = []
    obj_return_list = []
    util_return_list = []
    error_list = []
    constraint_violation_list = []
    for i in range(10):
        with tqdm(total=int(args.num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(args.num_episodes / 10)):
                phis_list = copy.deepcopy(old_policies)
                # old_agent is now updated agent
                old_policies = []
                for agent in agents:
                    old_policy = copy.deepcopy(agent.actors)
                    old_policies.append(old_policy)

                # episode_returns = 0
                episode_obj_returns = 0
                episode_util_returns = 0
                episode_constraint_violation = 0
                minibatch_disc_util_returns = []
                v_lists = []

                # consensus error
                errors = 0
                param_list = []
                for agent in agents:
                    params = []
                    for actor in agent.actors:
                        param = torch.nn.utils.convert_parameters.parameters_to_vector(actor.parameters()).detach()
                        params.append(param)
                    mparams = torch.cat([p.view(-1) for p in params]).detach()
                    param_list.append(mparams)
                for k in range(len(param_list)):
                    for t in range(len(param_list)):
                        errors += torch.norm(param_list[k] - param_list[t], 2)
                errors = errors / (num_agents ** 2)
                error_list.append(errors.cpu().numpy())

                for idx, (agent, env) in enumerate(zip(agents, envs)):
                    minibatch_v = []
                    # episode_return = 0
                    # episode_obj_return = 0
                    # episode_util_return = 0
                    episode_disc_util_returns = []
                    for b in range(args.minibatch_size):
                        # transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'obj_rewards': [], 'util_rewards': [], 'dones': []}
                        state = env.reset()
                        state = np.concatenate(state).ravel()
                        episode_disc_util_return = 0
                        for t in range(args.max_eps_len):
                            actions = agent.take_actions(state)
                            next_state, rewards, dones, _ = env.step(actions)
                            next_state = np.concatenate(next_state).ravel()
                            done = all(item == True for item in dones)
                            transition_dict['states'].append(state)
                            transition_dict['actions'].append(actions)
                            transition_dict['next_states'].append(next_state)
                            # transition_dict['rewards'].append(rewards[idx])
                            transition_dict['obj_rewards'].append(rewards[idx][0])
                            transition_dict['util_rewards'].append(rewards[idx][1])
                            transition_dict['dones'].append(dones[idx])
                            state = next_state
                            # episode_returns += np.sum(rewards[idx])
                            episode_obj_returns += rewards[idx][0]
                            episode_util_returns += rewards[idx][1]
                            episode_disc_util_return = episode_disc_util_return * args.gamma + rewards[idx][1]
                            reset = t == args.max_eps_len - 1
                            if done or reset:
                                break

                        episode_disc_util_returns.append(episode_disc_util_return)
                        episode_constraint_violation += args.constraint - episode_disc_util_return

                        advantages = agent.update_value(transition_dict)
                        single_traj_v = agent.compute_v_list(transition_dict, advantages, prev_v_lists[idx], phis_list[idx],
                                                          args.beta)
                        single_traj_v = [obj_grad_v + util_grad_v * agent.dual_param for obj_grad_v, util_grad_v in zip(*single_traj_v)]
                        single_traj_v = torch.stack(single_traj_v, dim=0)
                        minibatch_v.append(single_traj_v)

                    minibatch_disc_util_returns.append(np.mean(episode_disc_util_returns))

                    minibatch_v = torch.stack(minibatch_v, dim=0)
                    v_list = torch.mean(minibatch_v, dim=0)
                    v_lists.append(v_list)

                # return_list.append(episode_returns / args.minibatch_size)
                obj_return_list.append(episode_obj_returns / args.minibatch_size)
                util_return_list.append(episode_util_returns / args.minibatch_size)
                constraint_violation_list.append(episode_constraint_violation / args.minibatch_size)

                # Gradient tracking.
                y_lists = take_grad_consensus(y_lists, pi)
                next_y_lists = update_y_lists(y_lists, prev_v_lists, v_lists)

                # Take consensus for gradients and parameters.
                consensus_next_y_lists = take_grad_consensus(next_y_lists, pi)
                agents = take_param_consensus(agents, pi)

                # Update parameters.
                for agent, grads in zip(agents, consensus_next_y_lists):
                    update_param(agent, grads, lr=args.grad_lr, clip_grad=args.clip_grad)

                # Update dual parameters
                for agent, disc_util_return in zip(agents, minibatch_disc_util_returns):
                    agent.update_dual_param(disc_util_return, args.dual_lr, args.constraint, clip_grad=args.dual_clip_grad)

                y_lists = copy.deepcopy(next_y_lists)
                prev_v_lists = copy.deepcopy(v_lists)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (args.num_episodes / 10 * i + i_episode + 1),
                                      'obj_return': '%.3f' % np.mean(obj_return_list[-10:]),
                                      'util_return': '%.3f' % np.mean(util_return_list[-10:]),
                                      'constraint_violation': '%.3f' % np.mean(constraint_violation_list[-10:]),
                                      'dual_param': '%.3f' % np.mean([agent.dual_param.item() for agent in agents]),})
                    writer.add_scalar("obj_rewards", np.mean(obj_return_list[-10:]), args.num_episodes / 10 * i + i_episode + 1)
                    writer.add_scalar("util_rewards", np.mean(util_return_list[-10:]), args.num_episodes / 10 * i + i_episode + 1)
                    writer.add_scalar("constraint_violation", np.mean(constraint_violation_list[-10:]), args.num_episodes / 10 * i + i_episode + 1)
                    writer.add_scalar("dual_param", np.mean([agent.dual_param.item() for agent in agents]), args.num_episodes / 10 * i + i_episode + 1)
                pbar.update(1)
    # mv_return_list = moving_average(return_list, 9)
    return obj_return_list, util_return_list, error_list


if __name__ == '__main__':
    env_name = 'simple_spread_ac'
    training_args = [
        (2, 'ring', 0.2),
        (3, 'ring', 0.2),
        (4, 'ring', 0.2),
        (5, 'ring', 0.2),
        (2, 'dense', 0.2),
        (3, 'dense', 0.2),
        (4, 'dense', 0.2),
        (5, 'dense', 0.2),
        (2, 'bipartite', 0.2),
        (3, 'bipartite', 0.2),
        (4, 'bipartite', 0.2),
        (5, 'bipartite', 0.2),
        (6, 'ring', 0.2),
        (6, 'dense', 0.2),
        (6, 'bipartite', 0.2),
    ]
    # topologies = ['dense', 'ring', 'bipartite']
    # betas = [0.2, 0.2, 0.2]
    # labels = ['beta=0.2', 'beta=0.2', 'beta=0.2']
    # num_agents = 5

    for num_agents, topology, beta in training_args:
        print('-' * 60)
        print(f'Training {num_agents} agents in {topology} topology with beta={beta}')
        print('-' * 60)
        args = set_args(num_agents=num_agents, beta=beta, topology=topology)
        label = f'beta={beta}'
        fpath = os.path.join('mdpgt_results', env_name, str(num_agents) + '_agents',
                             label + '_' + topology)  # + '_' + timestr
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
        print(f"beta={beta}")

        obj_return_list, util_return_list, error_list = run(args=args, env_name=env_name)
        np.save(os.path.join(fpath, 'obj_return.npy'), obj_return_list)
        np.save(os.path.join(fpath, 'util_return.npy'), util_return_list)
        np.save(os.path.join(fpath, 'avg_obj_return.npy'), moving_average(obj_return_list, 9))
        np.save(os.path.join(fpath, 'avg_util_return.npy'), moving_average(util_return_list, 9))
        np.save(os.path.join(fpath, 'error.npy'), error_list)




