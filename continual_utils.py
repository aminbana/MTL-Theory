import torch

class Continual_Util:
    def __init__(self, num_classes_per_task, dataset_name, num_tasks, scenario):
        self.NUM_TOTAL_CLASSES = num_classes_per_task * num_tasks
        self.BENCHMARK = scenario #'task' or 'class'
        self.NUM_TASKS = num_tasks
        self.NUM_CLASSES_PER_TASK = self.NUM_TOTAL_CLASSES // self.NUM_TASKS
        self.task_id = 0
        self.dataset_name = dataset_name
        # print (f"{self.BENCHMARK} incremental continual learning scenario with {self.NUM_TASKS} tasks and {self.NUM_CLASSES_PER_TASK} classes in each task")


    def get_classes_for_task(self, task_id):
        return list(range(self.NUM_CLASSES_PER_TASK * task_id, self.NUM_CLASSES_PER_TASK * (task_id + 1)))

    def get_all_classes_before_task(self, task_id):
        return list(range(0, self.NUM_CLASSES_PER_TASK * (task_id + 1)))

    def modify_labels(self, y, ):
        if self.BENCHMARK in ['task', 'domain']:
            return y % self.NUM_CLASSES_PER_TASK, y // self.NUM_CLASSES_PER_TASK
        return y, y // self.NUM_CLASSES_PER_TASK

    def set_current_task(self, task_id):
        self.task_id = task_id

    def mask_task_labels(self, pred_vector, task_ids):
        return_tensor = []

        for i,_ in enumerate(pred_vector):
            start = task_ids[i] * self.NUM_CLASSES_PER_TASK
            end = start + self.NUM_CLASSES_PER_TASK
            return_tensor.append(pred_vector[i][start:end])
        return torch.stack (return_tensor)

    def mask_other_tasks(self, pred_vector, task_ids):
        return_tensor = []
        for i,_ in enumerate(pred_vector):
            start = task_ids[i] * self.NUM_CLASSES_PER_TASK
            end = start + self.NUM_CLASSES_PER_TASK

            mask = torch.zeros_like(pred_vector[i])
            mask[start:end] = 1.
            return_tensor.append(pred_vector[i] * mask)
        return torch.stack (return_tensor)
