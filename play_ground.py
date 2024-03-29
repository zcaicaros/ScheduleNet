import torch
from semiMDP.simulators import Simulator
import random
import numpy
import time

'''random.seed(0)
numpy.random.seed(1)
torch.manual_seed(1)
sim = Simulator(30, 30, verbose=False)
t1 = time.time()
while True:
    available_m_ids, aggregated_reward, done, _ = sim.flush_trivial_ops()
    if done:
        break

    for i, m_id in enumerate(available_m_ids):
        sim.transit()  # assign random action (operation)
        g, r, done = sim.observe(return_doable=True)

print(sim.global_time)
print(time.time() - t1)'''


random.seed(0)
numpy.random.seed(1)
torch.manual_seed(1)
sim = Simulator(3, 3, verbose=False, detach_done=True)
g, r, done = sim.observe()
print(g.number_of_edges())
sim.plot_graph()

print({1:2, 3:4}.items())