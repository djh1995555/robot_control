import numpy as np
from typing import List, Union

class Task:
    def __init__(self, name: str):
        self.task_name = name
        self.id = -1
        self.parent_id = -1
        self.child_id = -1
        self.dx_des = None
        self.ddx_des = None
        self.delta_q = None
        self.dq = None
        self.ddq = None
        self.J = None
        self.dJ = None
        self.J_pre = None
        self.N = None
        self.kp = None
        self.kd = None
        self.W = None  # Weighted matrix for pseudo inverse
        self.err_x = None
        self.derr_x = None

class PriorityTasks:
    def __init__(self):
        self.task_lib: List[Task] = []
        self.name_list: List[str] = []
        self.id_list: List[int] = []
        self.parent_id_list: List[int] = []
        self.child_id_list: List[int] = []
        self.out_delta_q = None
        self.out_dq = None
        self.out_ddq = None
        self.start_id = -1

    def add_task(self, name: str) -> None:
        """Add a new task to the task library"""
        new_task = Task(name)
        new_task.id = len(self.task_lib)
        self.task_lib.append(new_task)
        self.name_list.append(name)

    def get_id(self, name: Union[str, bytes]) -> int:
        """Get task ID by name"""
        if isinstance(name, bytes):
            name = name.decode('utf-8')
            
        for i, task_name in enumerate(self.name_list):
            if task_name == name:
                return i
                
        print(f"Cannot find wbc task: {name}!")
        return -1

    def build_priority(self, task_order: List[str]) -> None:
        """Build the priority hierarchy of tasks"""
        self.start_id = self.get_id(task_order[0])
        
        for i in range(len(task_order)):
            current_id = self.get_id(task_order[i])
            
            if i == 0:
                self.task_lib[current_id].parent_id = -1
            else:
                before_id = self.get_id(task_order[i-1])
                self.task_lib[current_id].parent_id = before_id
                
            if i == len(task_order) - 1:
                self.task_lib[current_id].child_id = -1
            else:
                next_id = self.get_id(task_order[i+1])
                self.task_lib[current_id].child_id = next_id

    def print_task_info(self) -> None:
        """Print information about all tasks"""
        for task in self.task_lib:
            print("-------------")
            print(f"task_name={task.task_name}")
            print(f"parent_id={task.parent_id}, child_id={task.child_id}")

    def compute_all(self, des_delta_q: np.ndarray, des_dq: np.ndarray, des_ddq: np.ndarray,
                   dyn_M: np.ndarray, dyn_M_inv: np.ndarray, dq: np.ndarray) -> None:
        """Compute all tasks in priority order"""
        current_id = self.start_id
        parent_id = self.task_lib[current_id].parent_id
        child_id = self.task_lib[current_id].child_id
        
        for _ in range(len(self.task_lib)):
            task = self.task_lib[current_id]
            
            if parent_id == -1:
                # First task in priority
                task.N = np.eye(task.J.shape[1])
                task.J_pre = task.J @ task.N
                
                # Compute weighted pseudo-inverse
                J_pinv = self.pseudo_inv_right_weighted(task.J_pre, task.W)
                task.delta_q = des_delta_q + J_pinv @ task.err_x
                task.dq = des_dq
                
                ddx_cmd = (task.ddx_des + task.kp @ task.err_x + task.kd @ task.derr_x)
                dyn_J_pinv = self.dyn_pseudo_inv(task.J_pre, dyn_M_inv, True)
                task.ddq = des_ddq + dyn_J_pinv @ (ddx_cmd - task.dJ @ dq)
            else:
                # Subsequent tasks
                parent_task = self.task_lib[parent_id]
                J_pinv_parent = self.pseudo_inv_right_weighted(parent_task.J_pre, parent_task.W)
                
                # Compute null space projection
                identity = np.eye(parent_task.J_pre.shape[1])
                task.N = parent_task.N @ (identity - J_pinv_parent @ parent_task.J_pre)
                task.J_pre = task.J @ task.N
                
                # Compute weighted pseudo-inverse for current task
                J_pinv = self.pseudo_inv_right_weighted(task.J_pre, task.W)
                
                # Update delta_q, dq, ddq
                task.delta_q = (parent_task.delta_q + 
                               J_pinv @ (task.err_x - task.J @ parent_task.delta_q))
                task.dq = (parent_task.dq + 
                          J_pinv @ (task.dx_des - task.J @ parent_task.dq))
                
                ddx_cmd = task.ddx_des + task.kp @ task.err_x + task.kd @ task.derr_x
                dyn_J_pinv = self.dyn_pseudo_inv(task.J_pre, dyn_M_inv, True)
                task.ddq = (parent_task.ddq + 
                           dyn_J_pinv @ (ddx_cmd - task.dJ @ dq - task.J @ parent_task.ddq))
            
            # Move to next task in priority chain
            if child_id != -1:
                parent_id = current_id
                current_id = child_id
                child_id = self.task_lib[current_id].child_id
            else:
                break
        
        # Store final outputs
        self.out_delta_q = self.task_lib[current_id].delta_q
        self.out_dq = self.task_lib[current_id].dq
        self.out_ddq = self.task_lib[current_id].ddq

    @staticmethod
    def pseudo_inv_right_weighted(J: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Compute weighted pseudoinverse of J with right weighting matrix W"""
        # Implementation depends on your specific pseudoinverse method
        # This is a placeholder - replace with your actual implementation
        return np.linalg.pinv(J)

    @staticmethod
    def dyn_pseudo_inv(J: np.ndarray, M_inv: np.ndarray, weighted: bool) -> np.ndarray:
        """Compute dynamic pseudoinverse of J"""
        # Implementation depends on your specific method
        # This is a placeholder - replace with your actual implementation
        return np.linalg.pinv(J)