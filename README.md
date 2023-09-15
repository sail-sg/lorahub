# LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition

The official repository which contains the code and pre-trained models for our paper [LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition](https://arxiv.org/abs/2307.13269).


# 🔥 Updates
- [**2023-9-13**]: Now Available for Easy Installation via `pip install lorahub`. For usage instructions regarding the interface, please refer to the `example.py` file
- [**2023-8-29**]: We released the full produce code at [reproduce_bbh.py](reproduce_bbh.py). Please checkout the script to reproduce our results!
- [**2023-8-03**]: Integrated into Replicate, check out the [demo](https://replicate.com/cjwbw/lorahub)!
- [**2023-7-27**]: We released our [code](https://github.com/sail-sg/lorahub) and [demo](https://huggingface.co/spaces/sail/lorahub). Check it out!
- [**2023-7-26**]: We released our [paper](https://arxiv.org/abs/2307.13269).


# 🏴󠁶󠁵󠁭󠁡󠁰󠁿 Overview

Low-rank adaptations (LoRA) are techniques for fine-tuning large language models on new tasks. We propose **LoraHub**, a framework that allows **composing multiple LoRA modules** trained on different tasks. The goal is to achieve good performance on **unseen tasks** using just a few examples, **without needing extra parameters or training**. And we want to build a marketplace where users can share their trained LoRA modules, thereby facilitating the application of these modules to new tasks.

<figure style="text-align:center">
  <img src="./figure/overview.jpg">
</figure>

The figure demostrates the zero-shot learning, few-shot in-context learning and few-shot lorahub learning (ours). Note that the Compose procedure is conducted per task rather than per example. Our method achieves similar inference throughput as zero-shot learning, yet approaches the performance of in-context learning on the BIG-Bench Hard (BBH) benchmark. The experimental results show the superior efficacy of our method in comparison to zero-shot learning while closely resembling the performance of in-context learning (ICL) in few-shot scenarios.

<br>

<figure style="text-align:center">
  <img src="./figure/pipeline.jpg">
</figure>

The figure shows the pipeline of LoraHub Learning. Our method encompasses two stages: the <strong>Compose</strong> stage and the <strong>Adapt</strong> stage. During the Compose stage, existing LoRA modules are integrated into one unified module, employing a set of weights, denoted as <em>w</em>, as coefficients. In the Adapt stage, the amalgamated LoRA module is evaluated on a few examples from the unseen task. Subsequently, a <strong>gradient-free algorithm</strong> is applied to refine <em>w</em>. After executing <em>K</em> iterations, a <strong>highly adapted LoRA module</strong> is produced, which can be incorporated with the LLM to perform the intended task.

<br>


# ⚡️ Quickstart

You can install lorahub using

```shell
pip install lorahub
```

And then you can use lorahub learning by simply calling the function `lorahub_learning`:

``` python
from lorahub.algorithm import lorahub_learning, lorahub_inference
from lorahub.constant import LORA_MODULE_NAMES
import random


def get_examples_for_learning():
    """
    Get a few examples to learn to compose given LoRA modules
    """
    return [
        {"input":
            "Infer the date from context.\n\nQ: Jane is celebrating the last day of Jan 2012. What is the date tomorrow in MM/DD/YYYY?\nOptions:\n(A) 02/02/2012\n(B) 02/15/2012\n(C) 01/25/2012\n(D) 04/22/2012\n(E) 02/01/2012\n(F) 02/11/2012\nA:", "output": "(E)"}
    ]

def get_lora_module_list():
    """
    You can have a custom filtering strategy to select the modules to be used in the composition. Here we randomly select 20 modules.
    """
    random.seed(42)
    return random.sample(LORA_MODULE_NAMES, 20)


# get a list of modules to be used in the composition
modules = get_lora_module_list()
print("modules:", modules)

# construct input list and output list
example_inputs, examples_outputs = [], []
for example in get_examples_for_learning():
    example_inputs.append(example["input"])
    examples_outputs.append(example["output"])

# perform LoRAHub learning
module_weights, model, tokenizer = lorahub_learning(lora_module_list=modules,
                                                    example_inputs=example_inputs,
                                                    example_outputs=examples_outputs,
                                                    max_inference_step=40,
                                                    batch_size=1)

print("module_weights:", module_weights)
```

The `lorahub_learning` function `lorahub_learning` has the following interface design:

```python
lorahub_learning(lora_module_list: List[str], # list of lora candidates
                 example_inputs: List[str],
                 example_outputs: List[str],
                 max_inference_step: int, 
                 model_name_or_path=None, # if not given, we will use the model_name_or_path in lora config
                 batch_size=None, 
                 get_loss=default_get_loss, # The function to get the objective for optimiztion, use loss as default (can be changed to something like acc. or similarity)
                 get_regular=default_l1_regularization,  # The function to get regularization term for the weight, use 0.05*|w_i| as default
                 seed=42)
```

A full example can be found in [example.py](example.py).

# 🌲 Project Structure

The lorahub source code is organized as below:

``` shell
|-- lorahub
    -- algorithm.py # main code for lorahub learning and inference
    -- constant.py # lora candidate module names
|-- example.py # usage code for demonstration purpose
```


# 🏰 Resource

## LoRA Candidates

Our methodology requires a compendium of LoRA modules trained on preceding tasks. For parity with Flan, we adopt the tasks utilized to instruct Flan-T5, thereby incorporating nearly 196 distinct tasks and their corresponding instructions via https://huggingface.co/datasets/conceptofmind/FLAN_2022. Following this, we created several LoRA modules as possible candidates. These LoRA modules can be accessed at https://huggingface.co/models?search=lorahub.

# 💬 Citation

If our work is useful for you, please consider citing our paper:

```bibtex
@misc{huang2023lorahub,
    title={LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition}, 
    author={Chengsong Huang and Qian Liu and Bill Yuchen Lin and Tianyu Pang and Chao Du and Min Lin},
    year={2023},
    eprint={2307.13269},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
