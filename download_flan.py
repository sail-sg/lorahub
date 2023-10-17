from datasets import load_dataset
import os

dataset_folder = "flan_task"

def download_flan():
    dataset = load_dataset("conceptofmind/FLAN_2022", split="train")
    # filter some examples from the dataset
    dataset = dataset.filter(lambda example: example['template_type'] == "zs_noopt", num_proc=32)
    # group the dataset using the task_name
    task_names = dataset.unique("task_name")
    for task_name in task_names:
        print("Processing task: ", task_name)
        # filter the dataset for the current task
        task_dataset = dataset.filter(lambda example: example['task_name'] == task_name, num_proc=32)
        # if the dataset is too large, we randomly sample 5000 examples for the training
        if len(task_dataset) > 10000:
            task_dataset = task_dataset.shuffle()
            task_dataset = task_dataset.select(range(10000))
        # save it into the task file
        task_name = task_name.replace("/", "_")
        task_dataset.to_json(os.path.join(dataset_folder, task_name + ".json"))

if __name__ == "__main__":
    download_flan()