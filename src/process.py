"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import torch
from src.env import create_train_env

from src.model import Mnih2016ActorCritic
AC_NN_MODEL = Mnih2016ActorCritic
ACTOR_HIDDEN_SIZE=256
CRITIC_HIDDEN_SIZE=256

import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from tensorboardX import SummaryWriter
import timeit

def local_train(index, opt, global_model, optimizer, save=False):
    torch.manual_seed(123 + index)
    if save:
        start_time = timeit.default_timer()
    writer = SummaryWriter(opt.log_path)
    #env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    env, num_states, num_actions = create_train_env(opt.layout)
    #local_model = ActorCritic(num_states, num_actions)
    local_model = AC_NN_MODEL(num_states, num_actions)
    if opt.use_gpu:
        local_model.cuda()
    local_model.train()
    state = torch.from_numpy(env.reset())
    if opt.use_gpu:
        state = state.cuda()
    done = True
    curr_step = 0
    curr_episode = 0
    while True:
        if save:
            # Save trained model at save_interval
            if curr_episode % opt.save_interval == 0 and curr_episode > 0:
                #torch.save(global_model.state_dict(),
                #           "{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
                torch.save(global_model.state_dict(),
                           "{}/gym-pacman_{}".format(opt.saved_path, opt.layout))
        print("Process {}. Episode {}".format(index, curr_episode))
        curr_episode += 1
        # Synchronize thread-specific parameters theta'=theta and theta'_v=theta_v
        local_model.load_state_dict(global_model.state_dict())
        # Reset gradients
        if done:
#             h_0 = torch.zeros((1, 512), dtype=torch.float)
#             c_0 = torch.zeros((1, 512), dtype=torch.float)
            h_0 = torch.zeros((1, ACTOR_HIDDEN_SIZE), dtype=torch.float)
            c_0 = torch.zeros((1, CRITIC_HIDDEN_SIZE), dtype=torch.float)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if opt.use_gpu:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        log_policies = []
        values = []
        rewards = []
        entropies = []
        # Local steps
        for _ in range(opt.num_local_steps):
            curr_step += 1
            # Model prediction from state. Returns two functions:
            # * Action prediction (Policy function) -> logits (array with every action-value)
            # * Value prediction (Value function)   -> value (single value state-value)
            logits, value, h_0, c_0 = local_model(state, h_0, c_0)
            # Softmax over action-values
            policy = F.softmax(logits, dim=1)
            # Log-softmax over action-values, to get the entropy of the policy
            log_policy = F.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)
            # From Async Methods for Deep RL:
            """ We also found that adding the entropy of the policy π to the
                objective function improved exploration by discouraging
                premature convergence to suboptimal deterministic poli-
                cies. This technique was originally proposed by (Williams
                & Peng, 1991), who found that it was particularly help-
                ful on tasks requiring hierarchical behavior."""
            # We sample one action given the policy probabilities
            m = Categorical(policy)
            action = m.sample().item()
            # Perform action_t according to policy pi
            # Receive reward r_t and new state s_t+1
            state, reward, done, _ = env.step(action)
            # render as seen for NN : 
            env.render(mode = 'human')
            state = torch.from_numpy(state)
            if opt.use_gpu:
                state = state.cuda()
            # If last local step, reset episode
            if curr_step > opt.num_global_steps:
                done = True
            if done:
                curr_step = 0
                state = torch.from_numpy(env.reset())
                if opt.use_gpu:
                    state = state.cuda()
            # Save state-value, log-policy, reward and entropy of
            # every state we visit, to gradient-descent later
            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)

            if done:
                # All local steps done.
                break
        # Initialize R/G_t: Discounted reward over local steps
        R = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            R = R.cuda()
        if not done:
            _, R, _, _ = local_model(state, h_0, c_0)

        gae = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            gae = gae.cuda()
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R
        # Gradiend descent over minibatch of local steps, from last to first step.
        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
            # Generalized Advantage Estimator (GAE)
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
            next_value = value
            # Accumulate discounted reward
            R = R * opt.gamma + reward
            # Accumulate gradients wrt parameters theta'
            actor_loss = actor_loss + log_policy * gae
            # Accumulate gradients wrt parameters theta'_v
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy
        # Total loss for 
        total_loss = -actor_loss + critic_loss - opt.beta * entropy_loss
        # TensorBoard
        writer.add_scalar("Train_{}/Loss".format(index), total_loss, curr_episode)
        optimizer.zero_grad()
        total_loss.backward()

        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            # global_param.grad is None
            global_param._grad = local_param.grad
        # Perform asynchronous update of theta and theta_v
        optimizer.step()

        if curr_episode == int(opt.num_global_steps / opt.num_local_steps):
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                print('The code runs for %.2f s ' % (end_time - start_time))
            return


def local_test(index, opt, global_model):
    torch.manual_seed(123 + index)
    #env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    env, num_states, num_actions = create_train_env(opt.layout)
    #local_model = ActorCritic(num_states, num_actions)
    local_model = AC_NN_MODEL(num_states, num_actions)
    local_model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    while True:
        curr_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())
        with torch.no_grad():
            if done:
#                 h_0 = torch.zeros((1, 512), dtype=torch.float)
#                 c_0 = torch.zeros((1, 512), dtype=torch.float)
                h_0 = torch.zeros((1, ACTOR_HIDDEN_SIZE), dtype=torch.float)
                c_0 = torch.zeros((1, CRITIC_HIDDEN_SIZE), dtype=torch.float)
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()

        logits, value, h_0, c_0 = local_model(state, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, _ = env.step(action)
        # render as seen for NN : 
        env.render()
        actions.append(action)

        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()
        state = torch.from_numpy(state)
