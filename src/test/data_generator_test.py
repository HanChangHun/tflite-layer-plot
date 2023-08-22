import json
from pathlib import Path
import unittest

from src.data_generator import DataGenerator, create_temp_dir


class TestDataGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestDataGenerator, cls).setUpClass()
        cls.model_path_1 = Path(
            "model/mobilenet_v2_1.0_224_inat_bird_quant.tflite"
        )
        cls.model_path_2 = Path("model/efficientnet-edgetpu-S_quant.tflite")
        cls.model_path_3 = Path("model/efficientnet-edgetpu-M_quant.tflite")
        cls.model_path_4 = Path("model/efficientnet-edgetpu-L_quant.tflite")
        cls.model_path_5 = Path(
            "model/tfhub_tf2_resnet_50_imagenet_ptq.tflite"
        )

    @classmethod
    def tearDownClass(cls):
        super(TestDataGenerator, cls).tearDownClass()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_set_up(self):
        pass

    def test_generate(self):
        for target_model in [
            self.model_path_3,
            self.model_path_2,
            self.model_path_4,
            self.model_path_5,
        ]:
            generator = DataGenerator(target_model)
            generator.generate()
            data = generator.get_data()
            print(data)

            # with open(Path("data") / (target_model.stem + ".json"), "w") as f:
            #     json.dump(data, f, indent=4)

    def test_get_activation_and_param(self):
        generator = DataGenerator(self.model_path_1)
        with create_temp_dir() as tmp_output_dir:
            activation, param_size = generator.get_activation_and_param(
                0, tmp_output_dir
            )
        print(activation, param_size)

    def test_get_cumulative_exec_time(self):
        generator = DataGenerator(self.model_path_1)
        with create_temp_dir() as tmp_output_dir:
            exec_time = generator.get_cumulative_exec_time(30, tmp_output_dir)
        print(exec_time)


if __name__ == "__main__":
    unittest.main()
