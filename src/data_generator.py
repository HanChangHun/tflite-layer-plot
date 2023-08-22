from pathlib import Path
import shutil
import subprocess
import tempfile


from edgetpu_utils.tflite_utils import (
    get_num_ops,
    get_output_tran_size,
    calculate_parameter_sizes,
)
from edgetpu_utils.partition import partition_with_layer_idxs
from edgetpu_utils.benchmark import benchmark_model


def create_temp_dir():
    tmp_par_dir = Path("./tmp/partition")
    tmp_par_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=tmp_par_dir))


class DataGenerator:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.num_ops = get_num_ops(self.model_path)
        self.data = {}

    def get_data(self):
        return self.data

    def generate(self):
        for i in range(self.num_ops):
            tmp_output_dir = create_temp_dir()
            activation, param_size = self.get_activation_and_param(
                i, tmp_output_dir
            )
            exec_time = self.get_cumulative_exec_time(i, tmp_output_dir)
            shutil.rmtree(tmp_output_dir)

            self.data[i] = {
                "activation": int(activation),
                "param_size": int(param_size),
                "exec_time": float(exec_time),
            }

    def partition_model(self, start_idx, end_idx, tmp_output_dir):
        return partition_with_layer_idxs(
            self.model_path, self.num_ops, start_idx, end_idx, tmp_output_dir
        )

    def get_activation_and_param(self, layer_idx, tmp_output_dir):
        segment_paths, seg_idx = self.partition_model(
            layer_idx, layer_idx, tmp_output_dir
        )
        activation = get_output_tran_size(segment_paths[seg_idx])
        param_size = calculate_parameter_sizes(segment_paths[seg_idx])
        return activation, param_size

    def get_cumulative_exec_time(self, layer_idx, tmp_output_dir):
        segment_paths, seg_idx = self.partition_model(
            0, layer_idx, tmp_output_dir
        )

        cmd = f"edgetpu_compiler -o {tmp_output_dir} {segment_paths[seg_idx]}"
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
        compiled_segment_path = tmp_output_dir / (
            Path(segment_paths[seg_idx]).stem + "_edgetpu.tflite"
        )
        segment_latency = benchmark_model(str(compiled_segment_path), 50)

        return segment_latency
