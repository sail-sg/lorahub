from lorahub.algorithm import lorahub_inference
import os
import json
from lorahub.algorithm import lorahub_learning, lorahub_inference
from lorahub.constant import LORA_MODULE_NAMES
import random
from random import shuffle


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


def evaluate_lorahub_results_few_shot(folder, flan_model_name):
    sub_dirs = os.listdir(folder)

    # 5 seeds used in our experiments
    for sub_dir in sub_dirs:
        # construct the few-shot examples for lorahub learning
        example_inputs, examples_outputs = [], []
        example_file_path = os.path.join(folder, sub_dir, "example.jsonl")
        for line in open(example_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            example_inputs.append(example["context"])
            examples_outputs.append(example["completion"])
            
        # random select 5 examples for each task
        shuffled_set = shuffle(zip(example_inputs, examples_outputs), seed=42)
        example_inputs, examples_outputs = zip(*shuffled_set)
        # take the first 5 examples
        example_inputs, examples_outputs = example_inputs[:5], examples_outputs[:5]

        # load the zero-shot examples for evaluation
        test_file_path = os.path.join(folder, sub_dir, "zero_shot.jsonl")
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            task_inputs.append(example["context"])
            task_outputs.append(example["completion"])

        task_perf_list = []
        for seed in range(1, 6):
            random.seed(seed)

            def get_lora_module_list():
                return random.sample(LORA_MODULE_NAMES, 20)
            # get a list of modules to be used in the composition
            modules = get_lora_module_list()

            # perform LoRAHub learning
            module_weights, model, tokenizer = lorahub_learning(lora_module_list=modules,
                                                                example_inputs=example_inputs,
                                                                example_outputs=examples_outputs,
                                                                max_inference_step=40,
                                                                batch_size=5)

            print("module_weights:", module_weights)

            """
            Perform inference to get predictions
            """
            _, task_acc = lorahub_inference(example_inputs=task_inputs,
                                            model_or_name_path=model,
                                            tokenizer_or_tokenizer_path=tokenizer,
                                            batch_size=10,
                                            # can set as None if you do not have the ground truth
                                            example_outputs=task_outputs)
            task_perf_list.append(task_acc)
        avg_perf, max_perf = sum(task_perf_list) / len(task_perf_list), max(task_perf_list)
        print("average perf:", avg_perf, "best perf:", max_perf)


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
    # five shot for lorahub models
    evaluate_lorahub_results_few_shot("data_bbh", "google/flan-t5-large")
