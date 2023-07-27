# LoraHub

Code will be released in several days. Please stay tuned.
# LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition


The official repository which contains the code and pre-trained models for our paper [LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition](https://arxiv.org/abs/2307.13269).



# 🔥 Updates

- [**2023-7-27**]: We released our [code](https://github.com/sail-sg/lorahub) and [demo page](https://huggingface.co/spaces/sail/lorahub). Check it out!


- [**2023-7-26**]: We released our [paper](https://arxiv.org/abs/2307.13269).


# 🏴󠁶󠁵󠁭󠁡󠁰󠁿 Overview
Low-rank adaptations (LoRA) are techniques for fine-tuning large language models on new tasks. We propose **LoraHub**, a framework that allows **composing multiple LoRA modules** trained on different tasks. The goal is to achieve good performance on **unseen tasks** using just a few examples, **without needing extra parameters or training**. And we want to build a marketplace where users can share their trained LoRA modules, thereby facilitating the application of these modules to new tasks.

<br>

<figure style="text-align:center">
  <img src="./figure/pipeline.jpg">
  <figcaption>Fig 1. The pipeline of LoraHub Learning. Our method encompasses two stages: the <strong>Compose</strong> stage and the <strong>Adapt</strong> stage. During the Compose stage, existing LoRA modules are integrated into one unified module, employing a set of weights, denoted as <em>w</em>, as coefficients. In the Adapt stage, the amalgamated LoRA module is evaluated on a few examples from the unseen task. Subsequently, a <strong>gradient-free algorithm</strong> is applied to refine <em>w</em>. After executing <em>K</em> iterations, a <strong>highly adapted LoRA module</strong> is produced, which can be incorporated with the LLM to perform the intended task.
</figure>


<br>

Our experimental data shows the superior efficacy of our method in comparison to zero-shot learning while closely resembling the performance of in-context learning (ICL) in few-shot scenarios.

<figure style="text-align:center">
  <img src="./figure/overview.jpg">
  <figcaption>Fig 2. The illustration of zero-shot learning, few-shot in-context learning and few-shot lorahub learning (ours). Note that the Compose procedure is conducted per task rather than per example. Our method achieves similar inference throughput as zero-shot learning, yet approaches the performance of in-context learning on the BIG-Bench Hard (BBH) benchmark.
</figure>

## Project

``` shell
|-- lorahub
    -- algorithm.py # mainly code for lorahub learning
    -- constant.py # base lora candidates
|-- example.py # simple demo code
```

To used LoraHub learning by calling function

``` python
def lorahub_learning(lora_module_list: List[str], # list of lora candidates
                     example_inputs: List[str],
                     example_outputs: List[str],
                     max_inference_step: int, 
                     model_name_or_path=None, # if not given, we will use the model_name_or_path in lora config
                     batch_size=None, 
                     get_loss=default_get_loss, # The function to get the objective for optimiztion, use loss as default (can be changed to something like acc. or similarity)
                     get_regular=default_l1_regularization,  # The function to get regularization term for the weight, use 0.05*|w_i| as default
                     seed=42)

```


# ⚡️ Quickstart

## Prepare Environment

First, you should run the following commands to install the latest lib developed for LoraHub.

```python
pip install datasets
pip install transformers
pip install peft
```

and some other requirements

```
pip install nevergrad
pip install torch
pip install tqdm
pip install pandas
pip install numpy
```

## Install LoraHub

The pypi package will be setup in few days.

# 🏰 Resource

## LoRA Candidates

Our methodology requires a compendium of LoRA modules trained on preceding tasks. For parity with Flan, we adopt the tasks utilized to instruct Flan-T5, thereby incorporating nearly $200$ distinct tasks and their corresponding instructions via https://huggingface.co/datasets/conceptofmind/FLAN\_2022. Following this, we created several LoRA modules as possible candidates. These LoRA modules can be accessed at https://huggingface.co/models?search=lorahub.


