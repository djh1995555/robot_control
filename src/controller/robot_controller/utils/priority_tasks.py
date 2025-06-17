import numpy as np
from src.utils.math import *
class Task():
    def __init__(self, task_name):
        self.task_name = task_name
        self.id = 0
        self.parent_id = 0
        self.child_id = 0
class PriorityTasks():
    def __init__(self,cfg):
        self.tasks = []
        self.task_names = []
    
    def add_task(self, task_name):
        task = Task(task_name)
        task.id = len(self.tasks)
        self.tasks.append(task)
        self.task_names.append(task_name)

    def get_task_id(self, task_name):
        for i in range(len(self.task_names)):
            if(self.task_names[i] == task_name):
                return i
        return -1

    def build_priority(self, task_names_order):
        self.start_id = self.get_task_id(task_names_order[0])
        for i in range(len(task_names_order)):
            if(i == 0):
                parent_task_id = -1
            else:
                parent_task_id = self.get_task_id(task_names_order[i - 1])      

            if(i == len(self.task_names_order)-1):
                child_task_id = -1
            else:
                child_task_id = self.get_task_id(task_names_order[i + 1])
            
            task_id = self.get_task_id(task_names_order[i])
            self.tasks[task_id].parent_id = parent_task_id
            self.tasks[task_id].child_id = child_task_id


    def compute_wbc_res(self, delta_q_des, dq_des, ddq_des, dyn_M, dyn_M_inv, dq):
        cur_id = self.start_id
        parent_id = self.tasks[cur_id].parent_id
        child_id = self.tasks[cur_id].child_id
        for i in range(len(self.tasks)):
            task = self.tasks[cur_id]
            if(parent_id == -1):
                task.N = np.identity(task.J.shape[1])
                task.Jpre = task.J @ task.N
                task.delta_q = delta_q_des + pseudoInv_right_weighted(task.Jpre, task.W) @ task.errX
                task.dq = dq_des
                ddxcmd = task.ddxDes + task.kp * task.errX + task.kd * task.derrX
                task.ddq = ddq_des + dyn_pseudoInv(task.Jpre, dyn_M_inv, True) @ (ddxcmd - task.dJ @ task.dq)
            else:
                parent_task = self.tasks[parent_id]
                I = np.identity(parent_task.Jpre.shape[1])
                task.N = parent_task.N @ (I - pseudoInv_right_weighted(parent_task.Jpre, parent_task.W) @ parent_task.Jpre)
                task.Jpre = task.J @ task.N
                task.delta_q = parent_task.delta_q + pseudoInv_right_weighted(task.Jpre, task.W) @ (
                    task.errX - task.J @ parent_task.delta_q
                )
                task.dq = parent_task.dq + pseudoInv_right_weighted(task.Jpre, task.W) @ (
                    task.dxDes - task.J @ parent_task.dq
                )
                ddxcmd = task.ddxDes + task.kp * task.errX + task.kd * task.derrX
                task.ddq = parent_task.ddq + dyn_pseudoInv(task.Jpre, dyn_M_inv, True) @ (
                    ddxcmd - task.dJ @ task.dq - task.J @ parent_task.ddq
                )
            if(child_id != -1):
                parent_id = cur_id
                cur_id = child_id
                child_id = self.task[cur_id].child_id
            else:
                break

        self.out_delta_q = self.tasks[cur_id].delta_q
        self.out_dq = self.tasks[cur_id].dq
        self.out_ddq = self.tasks[cur_id].ddq
