import torch
from semiMDP.simulators import Simulator
import random
import numpy
import time
from model import nx_to_pyg, TGA, Policy


def rollout(s, dev, embedding_net=None, policy_net=None, verbose=True):

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
                    policy_net is not None:  # network forward goes here
                idle_machine_and_its_doable_ops = s.get_doable_ops_in_dict()
                idle_machine, doable_op = list(idle_machine_and_its_doable_ops.keys()), list(idle_machine_and_its_doable_ops.values())
                idle_machine = [idx - 1 for idx in idle_machine]
                doable_op = [[idx + s.num_machine for idx in ops_m] for ops_m in doable_op]
                input_graphs = nx_to_pyg(g, dev)
                embedded_g = tga(**input_graphs)
                op_id, _ = policy(idle_machine, doable_op, embedded_g)
                s.transit(op_id - s.num_machine)
                p_list.append(op_id)
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

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    # dev = 'cpu'

    tga = TGA().to(dev)
    policy = Policy().to(dev)

    setting = 'm=10'  # 'm=10', 'j=40', 'free_for_all'
    detach = True

    if setting == 'm=10':
        j = [10, 15, 20, 25, 30, 35, 40]
        m = [10 for _ in range(len(j))]
    elif setting == 'j=40':
        m = [10, 15, 20, 25, 30, 35, 40]
        j = [40 for _ in range(len(m))]
    else:
        # j = [15, 20, 20, 30, 30, 50, 50, 100, 10, 20, 6, 10, 20]
        # m = [15, 15, 20, 15, 20, 15, 20, 20, 10, 15, 6, 10, 5]
        j = [10, 15, 20, 10, 15, 20, 30, 15, 20, 20, 50, 10, 20]
        m = [5, 5, 5, 10, 10, 10, 10, 15, 10, 15, 10, 10, 20]
    save_dir = 'plt/ScheduleNet_complexity_{}_reimplement_detach.npy'.format(setting) if detach else 'plt/ScheduleNet_complexity_{}_reimplement.npy'

    print('Warm start...')
    for p_m, p_j in zip([5], [5]):  # select problem size
        s = Simulator(p_m, p_j, verbose=False, detach_done=True)
        _, t, _ = rollout(s, dev, embedding_net=tga, policy_net=policy, verbose=False)

    times = []
    for p_m, p_j in zip(m, j):  # select problem size
        print('Problem size = (m={}, j={})'.format(p_m, p_j))
        s = Simulator(p_m, p_j, verbose=False, detach_done=detach)
        _, t, _ = rollout(s, dev, embedding_net=tga, policy_net=policy, verbose=True)
        times.append(t)

    numpy.save(save_dir, numpy.array(times))

