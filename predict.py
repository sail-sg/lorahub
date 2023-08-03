# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import shutil
import random
from typing import List
import torch
from cog import BasePredictor, Input, Path, BaseModel

from lorahub.algorithm import (
    lorahub_learning,
    default_get_loss,
    default_l1_regularization,
)
from lorahub.constant import LORA_MODULE_NAMES


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    def predict(
        self,
        example_inputs: str = Input(
            description="List of input examples, one Line one input.",
            default="Infer the date from context.  Q: Today, 8/3/1997, is a day that we will never forget. What is the date one week ago from today in MM/DD/YYYY? Options: (A) 03/27/1998 (B) 09/02/1997 (C) 07/27/1997 (D) 06/29/1997 (E) 07/27/1973 (F) 12/27/1997 A:\nInfer the date from context.  Q: May 6, 1992 is like yesterday to Jane, but that is actually ten years ago. What is the date tomorrow in MM/DD/YYYY? Options: (A) 04/16/2002 (B) 04/07/2003 (C) 05/07/2036 (D) 05/28/2002 (E) 05/07/2002 A:\nInfer the date from context.  Q: Today is the second day of the third month of 1966. What is the date one week ago from today in MM/DD/YYYY? Options: (A) 02/26/1966 (B) 01/13/1966 (C) 02/02/1966 (D) 10/23/1966 (E) 02/23/1968 (F) 02/23/1966 A:",
        ),
        example_outputs: str = Input(
            description="List of output examples, one Line one output.",
            default="(C)\n(E)\n(F)",
        ),
        lora_modules_specified: str = Input(
            description="Specify LoRA modules for the composition, options are from https://huggingface.co/models?search=lorahub, separated modules with comma, e.g. 'lorahub/flan_t5_large-quarel_logic_test, lorahub/flan_t5_large-coqa'",
            default=None,
        ),
        num_random_lora_modules: int = Input(
            description="Set number of LoRA modules to use. Ignored if specified modules above.",
            default=20,
            ge=2,
            le=196,
        ),
        max_inference_step: int = Input(
            description="Maximum iteration steps to maximise LoRA module composition. We suggest setting it to 40 steps if 20 modules were chosen, with more steps typically needed for more modules.",
            default=40,
            le=100,
            ge=10,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed.", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if lora_modules_specified:
            lora_module_list = lora_modules_specified.split(",")
            for lora_module in lora_module_list:
                assert (
                    lora_module in LORA_MODULE_NAMES
                ), f"{lora_module} is not recognised."
        else:
            lora_module_list = random.sample(LORA_MODULE_NAMES, num_random_lora_modules)

        example_inputs = example_inputs.splitlines()
        example_outputs = example_outputs.splitlines()
        assert len(example_inputs) == len(
            example_outputs
        ), "Number of input and output do not match."

        # perform LoRAHub learning
        module_weights, model, tokenizer = lorahub_learning(
            lora_module_list=lora_module_list,
            example_inputs=example_inputs,
            example_outputs=example_outputs,
            max_inference_step=max_inference_step,
            model_name_or_path=None,  # if not given, we will use the model_name_or_path in lora config
            batch_size=None,
            get_loss=default_get_loss,  # The function to get the objective for optimiztion, use loss as default (can be changed to something like acc. or similarity)
            get_regular=default_l1_regularization,  # The function to get regularization term for the weight, use 0.05*|w_i| as default
            seed=seed,
        )

        print("The recommended weight set for the LoRA modules is:")
        for module_weight, module in zip(module_weights, lora_module_list):
            print(f"{module_weight:.4f}: {module}")

        out = "/tmp/out.bin"
        torch.save(model, out)

        return Path(out)
