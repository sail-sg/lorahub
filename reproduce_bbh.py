from lorahub.algorithm import lorahub_inference
import os
import json

def evaluate_flan_results_zero_shot(folder, flan_model_name):
    sub_dirs = os.listdir(folder)
    
    for sub_dir in sub_dirs:
        test_file_path = os.path.join(folder, sub_dir, "zero_shot.jsonl")
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            task_inputs.append(example["context"])
            task_outputs.append(example["completion"])
        print("Evaluating on task (zero shot): ", sub_dir)
        lorahub_inference(task_inputs,
                        flan_model_name,
                        flan_model_name,
                        16,
                        task_outputs)

def evaluate_flan_results_few_shot(folder, flan_model_name):
    sub_dirs = os.listdir(folder)
    
    for sub_dir in sub_dirs:
        test_file_path = os.path.join(folder, sub_dir, "few_shot.jsonl")
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            task_inputs.append(example["context"])
            task_outputs.append(example["completion"])
        print("Evaluating on task (five shot): ", sub_dir)
        lorahub_inference(task_inputs,
                        flan_model_name,
                        flan_model_name,
                        16,
                        task_outputs)
    

if __name__ == "__main__":
    if not os.path.exists("data_bbh"):
        # download dataset
        os.system("wget https://github.com/sail-sg/lorahub/releases/download/0.1/data_bbh.zip")
        # unzip
        os.system("unzip data_bbh.zip")
    # evaluate the model
    evaluate_flan_results_zero_shot("data_bbh", "google/flan-t5-large")
    # five shot for flan models
    evaluate_flan_results_few_shot("data_bbh", "google/flan-t5-large")
