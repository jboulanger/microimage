from microimage.readers import CZIReader
import numpy as np
from pathlib import Path


def test_czi_reader():
    reader = CZIReader(Path("data/111125_GFP 488_PCNT 647_SunBicD2 FL CCmut new_1.czi"))

    for sc in reader.iter_scenes():
        assert sc.shape == (1, 13, 3, 1, 2101, 2101)

    reader = CZIReader(Path("data/Experiment-360.czi"))
    for sc in reader.iter_scenes():
        print(sc.shape)
