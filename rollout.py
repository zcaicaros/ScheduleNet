import torch
from pyjssp.simulators import Simulator
import random
import numpy
import time


def rollout(s, dev, embedding_net=None, policy_net=None, critic_net=None, verbose=True):

    if embedding_net is not None and \
            policy_net is not None and \
            critic_net is not None:
        embedding_net.to(dev)
        policy_net.to(dev)
        critic_net.to(dev)

    s.reset()
    done = False

    p_list = []
    t1 = time.time()
    while True:
        do_op_dict = s.get_doable_ops_in_dict()
        all_machine_work = False if bool(do_op_dict) else True

        if all_machine_work:  # all machines are on processing. keep process!
            s.process_one_time()
        else:  # some of machine has possibly trivial action. the others not.
            _, _, done, sub_list = s.flush_trivial_ops(reward='makespan')  # flush the trivial action
            p_list += sub_list
            if done:
                break  # env rollout finish
            g, r, done = s.observe(return_doable=True)
            if embedding_net is not None and \
                    policy_net is not None and \
                    critic_net is not None:  # network forward goes here
                pass
            else:
                op_id = s.transit()
                p_list.append(op_id)

        if done:
            break  # env rollout finish
    t2 = time.time()
    if verbose:
        print('All job finished, makespan={}. Rollout takes {} seconds'.format(s.global_time, t2 - t1))
    return p_list, t2 - t1, s.global_time


if __name__ == "__main__":
    random.seed(0)
    numpy.random.seed(1)
    torch.manual_seed(1)

    # j = [5+5*i for i in range(6)]
    # m = [5 for _ in range(len(j))]

    # m = [5+5*i for i in range(6)]
    # j = [30 for _ in range(len(m))]

    m = [30]
    j = [30]

    print('Warm start...')
    for p_m, p_j in zip([5], [5]):  # select problem size
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        # dev = 'cpu'
        s = Simulator(p_m, p_j, verbose=False)
        _, t, _ = rollout(s, dev, verbose=False)
    times = []
    for p_m, p_j in zip(m, j):  # select problem size
        print('Problem size = (m={}, j={})'.format(p_m, p_j))
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        # dev = 'cpu'
        s = Simulator(p_m, p_j, verbose=False)
        _, t, _ = rollout(s, dev)
        times.append(t)

    # print(times)

