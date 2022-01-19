import random
from collections import OrderedDict
import numpy as np
from semiMDP.configs import (
    NOT_START_NODE_SIG,
    PROCESSING_NODE_SIG,
    DONE_NODE_SIG,
    DELAYED_NODE_SIG
)
import networkx as nx


class MachineManager:
    def __init__(self,
                 machine_matrix,
                 job_manager,
                 delay=True,  # True: prev op is processing, next op is processable
                 verbose=False):

        machine_matrix = machine_matrix.astype(int)
        self.job_manager = job_manager

        # Parse machine indices
        machine_index = list(set(machine_matrix.flatten().tolist()))

        # Global machines dict
        self.machines = OrderedDict()
        for m_id in machine_index:
            job_ids, step_ids = np.where(machine_matrix == m_id)
            possible_ops = []
            for job_id, step_id in zip(job_ids, step_ids):
                possible_ops.append(job_manager[job_id][step_id])
            m_id += 1  # To make machine index starts from 1
            self.machines[m_id] = Machine(m_id, possible_ops, delay, verbose)

    def do_processing(self, t):
        for _, machine in self.machines.items():
            machine.do_processing(t)

    def load_op(self, machine_id, op, t):
        self.machines[machine_id].load_op(op, t)

    def __getitem__(self, index):
        return self.machines[index]

    # available: have remaining ops, idle, and not waiting for delayed op, i.e. those can be assigned
    def get_available_machines(self, shuffle_machine=True):
        m_list = []
        for _, m in self.machines.items():
            if m.available():
                m_list.append(m)

        if shuffle_machine:
            m_list = random.sample(m_list, len(m_list))

        return m_list
    
    # get idle machines' list
    def get_idle_machines(self):
        m_list = []
        for _, m in self.machines.items():
            if m.current_op is None and not m.work_done():
                m_list.append(m)
        return m_list
    
    # calculate the length of queues for all machines
    def cal_total_cost(self):
        c = 0
        for _, m in self.machines.items():
            c += len(m.doable_ops_no_delay)  # number of ready operations of m
        return c

    # update all cost functions of machines
    def update_cost_function(self, cost):
        for _, m in self.machines.items():
            m.cost += cost

    def get_machines(self):
        m_list = [m for _, m in self.machines.items()]
        return random.sample(m_list, len(m_list))

    def all_delayed(self):
        return np.product([m.delayed_op is not None for _, m in self.machines.items()])

    def fab_stuck(self):
        # All machines are not available and All machines are delayed.
        all_machines_not_available_cond = not self.get_available_machines()
        all_machines_delayed_cond = self.all_delayed()
        return all_machines_not_available_cond and all_machines_delayed_cond

    def observe(self, detach_done=True):
        """
        generate graph representation
        :return: nx.OrderedDiGraph
        """

        num_machine = len(self.machines)

        # create agents clique
        target_agents = self.get_available_machines(shuffle_machine=False)  # target agents are those idle and non-waiting
        g = nx.DiGraph()
        for m_id, m in self.machines.items():  # add node
            _x_machine = OrderedDict()
            _x_machine['agent'] = 1
            _x_machine['target_agent'] = 1 if m in target_agents else 0
            _x_machine['assigned'] = 1 - int(m.current_op is None)
            _x_machine['waiting'] = int(m.wait_for_delayed())
            _x_machine['processable'] = 0  # flag for operation node
            _x_machine['accessible'] = 0  # flag for operation node
            # features = -1 for machine node
            _x_machine['task_wait_time'] = m.delayed_op.wait_time if m.delayed_op is not None else -1
            _x_machine['task_processing_time'] = m.current_op.processing_time if m.current_op is not None else -1
            _x_machine['time_to_complete'] = m.remaining_time
            _x_machine['remain_ops'] = len(m.remain_ops)
            _x_machine['job_completion_ratio'] = m.current_op.complete_ratio if m.current_op is not None else -1
            # node type
            _x_machine['node_type'] = 'assigned_agent' if _x_machine['assigned'] == 1 else 'unassigned_agent'
            g.add_node(m_id - 1, **_x_machine)  # machine id from 0
            for neighbour_machine in self.machines.keys():  # fully connect to other machines
                if neighbour_machine != m_id:
                    g.add_edge(m_id - 1, neighbour_machine - 1, edge_feature=[0])  # edge_feature = not processable by the source node

        # create task subgraph
        for job_id, job in self.job_manager.jobs.items():
            for op in job.ops:
                not_start_cond = (op.node_status == NOT_START_NODE_SIG)
                delayed_cond = (op.node_status == DELAYED_NODE_SIG)
                processing_cond = (op.node_status == PROCESSING_NODE_SIG)
                done_cond = (op.node_status == DONE_NODE_SIG)

                if not_start_cond:
                    _x_task = OrderedDict()
                    _x_task['id'] = op._id
                    _x_task["type"] = op.node_status
                    _x_task["job_completion_ratio"] = op.complete_ratio
                    _x_task['task_processing_time'] = op.processing_time
                    _x_task['remain_ops'] = op.remaining_ops
                    _x_task['task_wait_time'] = op.waiting_time
                    _x_task["time_to_complete"] = -1
                    # ScheduleNet feature
                    _x_task["agent"] = 0
                    _x_task["target_agent"] = 0
                    _x_task["assigned"] = 0  # not_start_cond = op not load, i.e. not assigned
                    _x_task["waiting"] = 0
                    processable = int(op in self.machines[op.machine_id].doable_ops() and self.machines[op.machine_id] in target_agents)
                    _x_task["processable"] = processable
                    _x_task["accessible"] = processable * int(self.machines[op.machine_id].status())
                    _x_task['node_type'] = 'processable_task' if processable == 1 else 'unprocessable_task'
                elif processing_cond:
                    _x_task = OrderedDict()
                    _x_task['id'] = op._id
                    _x_task["type"] = op.node_status
                    _x_task["job_completion_ratio"] = op.complete_ratio
                    _x_task['task_processing_time'] = op.processing_time
                    _x_task['remain_ops'] = op.remaining_ops
                    _x_task['task_wait_time'] = 0
                    _x_task["time_to_complete"] = op.remaining_time
                    # ScheduleNet feature
                    _x_task["agent"] = 0
                    _x_task["target_agent"] = 0
                    _x_task["assigned"] = 1
                    _x_task["waiting"] = 0
                    _x_task["processable"] = 0
                    _x_task["accessible"] = 0
                    _x_task['node_type'] = 'assigned_task'
                elif done_cond:
                    _x_task = OrderedDict()
                    _x_task['id'] = op._id
                    _x_task["type"] = op.node_status
                    _x_task["job_completion_ratio"] = op.complete_ratio
                    _x_task['task_processing_time'] = op.processing_time
                    _x_task['remain_ops'] = op.remaining_ops
                    _x_task['task_wait_time'] = 0
                    _x_task["time_to_complete"] = -1
                    # ScheduleNet feature
                    _x_task["agent"] = 0
                    _x_task["target_agent"] = 0
                    _x_task["assigned"] = 1
                    _x_task["waiting"] = 0
                    _x_task["processable"] = 0
                    _x_task["accessible"] = 0
                    _x_task['node_type'] = 'completed_task'
                elif delayed_cond:
                    raise NotImplementedError("delayed operation")
                else:
                    raise RuntimeError("Not supporting node type")

                done_cond = _x_task["type"] == DONE_NODE_SIG

                node_id = op.id + num_machine  # task node iterate from num_machine + i
                g.add_node(node_id, **_x_task, task_done=done_cond)
                if detach_done:
                    if not done_cond:
                        g.add_edge(node_id, op.machine_id - 1, edge_feature=[0])  # task node -> agent node
                        machine_to_task_arc_feature = int(op in self.machines[op.machine_id].doable_ops())
                        g.add_edge(op.machine_id - 1, node_id, edge_feature=[machine_to_task_arc_feature])  # agent node -> task node
                        # out degrees for this op in job clique
                        for op_con in op.conjunctive_ops:
                            if op_con.node_status != DONE_NODE_SIG:
                                node_id_op_con = op_con.id + num_machine
                                g.add_edge(node_id, node_id_op_con, edge_feature=[0])
                else:
                    node_id = op.id + num_machine  # task node iterate from num_machine + i
                    g.add_node(node_id, **_x_task)
                    g.add_edge(node_id, op.machine_id - 1, edge_feature=[0])  # task node -> agent node
                    machine_to_task_arc_feature = int(op in self.machines[op.machine_id].doable_ops())
                    g.add_edge(op.machine_id - 1, node_id, edge_feature=[machine_to_task_arc_feature])  # agent node -> task node
                    # out degrees for this op in job clique
                    for op_con in op.conjunctive_ops:
                        node_id_op_con = op_con.id + num_machine
                        g.add_edge(node_id, node_id_op_con, edge_feature=[0])

        return g


class Machine:
    def __init__(self, machine_id, possible_ops, delay, verbose):
        self.machine_id = machine_id
        self.possible_ops = possible_ops
        self.remain_ops = possible_ops
        self.current_op = None
        self.delayed_op = None
        self.prev_op = None
        self.remaining_time = 0
        self.done_ops = []
        self.num_done_ops = 0
        self.cost = 0
        self.delay = delay
        self.verbose = verbose

    def __str__(self):
        return "Machine {}".format(self.machine_id)

    def status(self):
        currently_not_processing_cond = self.current_op is None
        not_wait_for_delayed_cond = not self.wait_for_delayed()
        status = currently_not_processing_cond and not_wait_for_delayed_cond
        return status

    def available(self):
        future_work_exist_cond = bool(self.doable_ops())
        currently_not_processing_cond = self.current_op is None
        not_wait_for_delayed_cond = not self.wait_for_delayed()
        ret = future_work_exist_cond and currently_not_processing_cond and not_wait_for_delayed_cond
        return ret

    def wait_for_delayed(self):
        wait_for_delayed_cond = self.delayed_op is not None
        ret = wait_for_delayed_cond
        if wait_for_delayed_cond:
            delayed_op_ready_cond = self.delayed_op.prev_op.node_status == DONE_NODE_SIG
            ret = ret and not delayed_op_ready_cond
        return ret

    def doable_ops(self):
        # doable_ops are subset of remain_ops.
        # some ops are doable when the prev_op is 'done' or 'processing' or 'start'
        doable_ops = []
        for op in self.remain_ops:
            prev_start = op.prev_op is None
            if prev_start:
                doable_ops.append(op)
            else:
                prev_done = op.prev_op.node_status == DONE_NODE_SIG
                prev_process = op.prev_op.node_status == PROCESSING_NODE_SIG
                first_op = not bool(self.done_ops)
                if self.delay:
                    # each machine's first processing operation should not be a reserved operation
                    if first_op:
                        cond = prev_done
                    else:
                        cond = (prev_done or prev_process)
                else:
                    cond = prev_done

                if cond:
                    doable_ops.append(op)
                else:
                    pass

        return doable_ops

    @property
    def doable_ops_id(self):
        doable_ops_id = []
        doable_ops = self.doable_ops()
        for op in doable_ops:
            doable_ops_id.append(op.id)

        return doable_ops_id

    @property
    def doable_ops_no_delay(self):
        doable_ops = []
        for op in self.remain_ops:
            prev_start = op.prev_op is None
            if prev_start:
                doable_ops.append(op)
            else:
                prev_done = op.prev_op.node_status == DONE_NODE_SIG 
                if prev_done:
                    doable_ops.append(op)
        return doable_ops

    def work_done(self):
        return not self.remain_ops

    def load_op(self, t, op):

        # Procedures for double-checkings
        # If machine waits for the delayed job is done:
        if self.wait_for_delayed():
            raise RuntimeError("Machine {} waits for the delayed job {} but load {}".format(self.machine_id,
                                                                                  print(self.delayed_op), print(op)))

        # ignore input when the machine is not available
        if not self.available():
            raise RuntimeError("Machine {} is not available".format(self.machine_id))

        # ignore when input op's previous op is not done yet:
        if not op.processable():
            raise RuntimeError("Operation {} is not accessible yet".format(print(op)))

        if op not in self.possible_ops:
            raise RuntimeError("Machine {} can't perform ops {}{}".format(self.machine_id,
                                                                          op.job_id,
                                                                          op.step_id))

        # Essential condition for checking whether input is delayed
        # if delayed then, flush dealed_op attr
        if op == self.delayed_op:
            if self.verbose:
                print("[DELAYED OP LOADED] / MACHINE {} / {} / at {}".format(self.machine_id, op, t))
            self.delayed_op = None

        else:
            if self.verbose:
                print("[LOAD] / Machine {} / {} on at {}".format(self.machine_id, op, t))

        # Update operation's attributes
        op.node_status = PROCESSING_NODE_SIG
        op.remaining_time = op.processing_time
        op.start_time = t

        # Update machine's attributes
        self.current_op = op
        self.remaining_time = op.processing_time
        self.remain_ops.remove(self.current_op)

    def unload(self, t):
        if self.verbose:
            print("[UNLOAD] / Machine {} / Op {} / t = {}".format(self.machine_id, self.current_op, t))
        self.current_op.node_status = DONE_NODE_SIG
        self.current_op.end_time = t
        self.done_ops.append(self.current_op)
        self.num_done_ops += 1
        self.prev_op = self.current_op
        self.current_op = None
        self.remaining_time = 0

    def do_processing(self, t):
        if self.remaining_time > 0:  # When machine do some operation
            if self.current_op is not None:
                self.current_op.remaining_time -= 1
                if self.current_op.remaining_time <= 0:
                    if self.current_op.remaining_time < 0:
                        raise RuntimeWarning("Negative remaining time observed")
                    if self.verbose:
                        print("[OP DONE] : / Machine  {} / Op {}/ t = {} ".format(self.machine_id, self.current_op, t))
                    self.unload(t)
            # to compute idle_time reward, we need to count delayed_time
            elif self.delayed_op is not None:
                self.delayed_op.delayed_time += 1
                self.delayed_op.remaining_time -= 1

            doable_ops = self.doable_ops()
            if doable_ops:
                for op in doable_ops:
                    op.waiting_time += 1
            else:
                pass

            self.remaining_time -= 1

    def transit(self, t, a):
        if self.available():  # Machine is ready to process.
            if a.processable():  # selected action is ready to be loaded right now.
                self.load_op(t, a)
            else:  # When input operation turns out to be 'delayed'
                a.node_status = DELAYED_NODE_SIG
                self.delayed_op = a
                self.delayed_op.remaining_time = a.processing_time + a.prev_op.remaining_time
                self.remaining_time = a.processing_time + a.prev_op.remaining_time
                self.current_op = None  # MACHINE is now waiting for delayed ops
                if self.verbose:
                    print("[DELAYED OP CHOSEN] : / Machine  {} / Op {}/ t = {} ".format(self.machine_id, self.delayed_op, t))
        else:
            raise RuntimeError("Access to not available machine")


if __name__ == "__main__":
    from semiMDP.jobShopSamplers import jssp_sampling
    from operationHelpers import JobManager

    random.seed(0)
    np.random.seed(1)

    ms, prts = jssp_sampling(3, 3)
    job_manager = JobManager(ms, prts, use_surrogate_index=True)
    machine_manager = MachineManager(ms, job_manager, delay=True, verbose=False)

    g = machine_manager.observe()
    for n in g.nodes:
        print(n, g.nodes[n])
