"""This module mainly contains the End2EndProcessor class. This class is intended
to unify the full processing pipeline from input to output. IT however is not intended
to be used for exporting or visualization. Depending on the use case it might be
useful to add more converters to the End2EndProcessor class.
"""
from typing import Dict, Any, Callable, List, Tuple
from copy import deepcopy
from pathlib import Path

import torch
from matplotlib import pyplot as plt
import yaml


from contact_grasp_net.dataloading.datasets import YCBSimulationData
from contact_grasp_net.datatypes import YCBSimulationDataSample
from contact_grasp_net.preprocessing import Preprocessor, PreprocessorBase
from contact_grasp_net.postprocessing import Postprocessor, PostprocessorBase
from contact_grasp_net.models import MyModel
from contact_grasp_net.utils import visualization as vis
from contact_grasp_net.utils.export import Exporter
from contact_grasp_net.utils.config import module_from_config


class End2EndProcessor:
    def __init__(
        self,
        model: MyModel = None,
        preprocessor: PreprocessorBase = None,
        postprocessor: PostprocessorBase = None,
        # more converters here
    ):
        """This is a utility class that combines the preprocessing, inference and
        postprocessing steps into a single callable object. While executing the
        different steps, it also collects intermediate results and stores them.
        These results can be used for debugging or visualization purposes.
        The input of the call is a list of samples, which are processed in a
        batched manner. The output is a list of dictionaries, where each dictionary
        contains the results of the processing steps for a single sample.
        Note that this used no dataloader to since the dataloader is not
        able to output the intermediate results in the preprocessing steps.
        Therfore this module only is for evaluation and debugging purposes and not
        for training.

        Args:
            model (GenerativeResnet, optional): _description_. Defaults to None.
            preprocessor (PreprocessorBase, optional): _description_. Defaults to None.
            postprocessor (PostprocessorBase, optional): _description_. Defaults to None.
        """
        self.model = model or MyModel.from_jit()
        self.preprocessor = preprocessor or Preprocessor()
        self.postprocessor = postprocessor or Postprocessor()
        # more converters here

    def _batched_processing(
        self, samples: List[Any], func: Callable
    ) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """Takes a list of samples and a function and applies the function to
        each sample in the list. The function is assumed to have an attribute
        "intermediate_results" which is a dictionary containing the intermediate
        results of the function. These results are collected and returned.

        Args:
            samples (List[Any]): The list of samples.
            func (Callable): The function to be applied to each sample.

        Returns:
            Tuple[List[Any], List[Dict[str, Any]]]: The output of the function 
            for each sample and the intermediate results.
        """
        output_batch = []
        intermediate_results = []
        for sample in samples:
            output_batch.append(func(sample))
            intermediate_results.append(deepcopy(func.intermediate_results))

        return output_batch, intermediate_results

    def __call__(self, samples: List[YCBSimulationDataSample]) -> List[Dict[str, Any]]:
        """The call method of the End2EndProcessor class. It takes a list of
        samples and processes them in a batched manner. The intermediate results
        are collected and returned.

        Args:
            samples (List[YCBSimulationDataSample]): The list of samples.

        Returns:
            List[Dict[str, Any]]: A collection of intermediate results for each
                sample and processing step.
        """

        # batched preprocessing
        input_batch, preprocessor_results = self._batched_processing(
            samples, self.preprocessor
        )
        input_batch = torch.stack(input_batch)

        # batched inference
        with torch.no_grad():
            predictions_batch = self.model(input_batch)

        # batched postprocessing
        postprocessed_batch, posprocessor_results = self._batched_processing(
            predictions_batch, self.postprocessor
        )

        # TODO: further postprocessing here

        # batched data collection
        process_data_batch = []
        for i_sample in range(len(samples)):
            process_data = {
                "preprocessor": preprocessor_results[i_sample],
                "postprocessor": posprocessor_results[i_sample],
                "model_input": input_batch[i_sample],
                "sample": samples[i_sample],
                "postprocessed": postprocessed_batch[i_sample],
            }
            process_data_batch.append(process_data)

        return process_data_batch


def process_dataset(
    dataset: YCBSimulationData,
    e2e_processor: End2EndProcessor,
    exporter: Exporter,
    batch_size=10,
):
    for i_batch in range((len(dataset) // batch_size) + 1):
        j_start = i_batch * batch_size
        j_end = min((i_batch + 1) * batch_size, len(dataset))
        batch = [dataset[j] for j in range(j_start, j_end)]
        print(f"Processing samples {j_start}...{j_end-1}")

        process_data_batch = e2e_processor(batch)

        for process_data in process_data_batch:
            fig = vis.overview_fig(
                fig=plt.figure(figsize=(20, 20)),
                # TODO more args here
            )
            plt.close(fig)

            export_data = {
                # "sample_attr": process_data["sample"].attr,
                # "result_attr": process_data["postprocessed"].attr,
                # more data here
                "overview": fig,
            }

            _ = exporter(export_data, f"{process_data['sample'].name}")


def process_YCBSimulationData(
    dataset_path: Path, config_path: Path, export_path: Path, batch_size: int
):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset = YCBSimulationData(dataset_path)

    e2e_processor = module_from_config(config)

    exporter = Exporter(export_dir=export_path)

    export_path.mkdir(parents=True, exist_ok=True)
    with open(export_path / "default_e2e_inference.yaml", "w") as f:
        yaml.dump(config, f)

    process_dataset(dataset, e2e_processor, exporter, batch_size)
