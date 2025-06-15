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

    def get_takk_id(self, task_name):
        for i in range(len(self.task_names)):
            if(self.task_names[i] == task_name):
                return i
        return -1

    def build_priority(self, task_names_order):
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


    def compute_wbc_res(task_name):
        None