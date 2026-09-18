import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
import xarray as xr

from microimage.manager import ExperimentManager
from microimage.models import DataItem
from microimage.processor import Processor, apply
from microimage.results import Mask, Table


class DummyProcessor(Processor):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def validate(
        self, item: DataItem, data: xr.DataArray
    ) -> Tuple[bool, Optional[str]]:
        if "RNA" not in data.coords.get("C", []):
            return False, "Missing 'RNA' channel"
        return True, None

    def get_parameters(self) -> Optional[Dict[str, Any]]:
        return {"threshold": self.threshold}

    def process(self, item: DataItem, data: xr.DataArray) -> Dict[str, Any]:
        return {
            "mask": Mask((data > self.threshold).astype(np.uint8)),
            "stats": Table([{"item_id": item.id, "mean": float(data.mean())}]),
        }


def test_processor_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        data_dir = tmp_path / "data"
        results_dir = tmp_path / "results"
        db_path = tmp_path / "test.db"

        data_dir.mkdir()
        results_dir.mkdir()

        manager = ExperimentManager(
            data_dir=data_dir,
            results_dir=results_dir,
            db_path=db_path,
        )

        item = DataItem(
            id="item1:0",
            relative_path=Path("dummy.tif"),
            scene="0",
        )
        item.channels = ["RNA"]
        item.size_x = 10
        item.size_y = 10
        item.size_c = 1
        item.scale_x = 1.0
        item.scale_y = 1.0

        conn = manager._get_connection()
        manager._create_item_in_db(conn, item)
        conn.commit()
        conn.close()

        # Mock image data
        data = xr.DataArray(
            np.ones((1, 10, 10)),
            dims=["C", "Y", "X"],
            coords={"C": ["RNA"]},
        )

        processor = DummyProcessor(threshold=0.5)

        # Apply with pre-loaded tuple (item, data)
        res = apply(manager, processor, items=[(item, data)])
        assert len(res) == 1
        item_id, success, err = res[0]
        assert success is True
        assert err is None

        # Check DB updated
        updated_item = manager["item1:0"]
        assert updated_item.processed is True
        assert updated_item.error is None
        assert "mask" in updated_item.result_files
        assert "stats" in updated_item.result_files
        assert "parameters" in updated_item.result_files

        # Check files exist
        assert (results_dir / updated_item.result_files["mask"]["path"]).exists()
        assert (results_dir / updated_item.result_files["stats"]["path"]).exists()
        assert (results_dir / updated_item.result_files["parameters"]["path"]).exists()


def test_processor_validation_failure():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        data_dir = tmp_path / "data"
        results_dir = tmp_path / "results"
        db_path = tmp_path / "test.db"

        data_dir.mkdir()
        results_dir.mkdir()

        manager = ExperimentManager(
            data_dir=data_dir,
            results_dir=results_dir,
            db_path=db_path,
        )

        item = DataItem(
            id="item2:0",
            relative_path=Path("dummy2.tif"),
            scene="0",
        )
        item.channels = ["DAPI"]
        item.size_x = 10
        item.size_y = 10
        item.size_c = 1

        conn = manager._get_connection()
        manager._create_item_in_db(conn, item)
        conn.commit()
        conn.close()

        # Image without "RNA" channel
        data = xr.DataArray(
            np.ones((1, 10, 10)),
            dims=["C", "Y", "X"],
            coords={"C": ["DAPI"]},
        )

        processor = DummyProcessor()
        res = apply(manager, processor, items=[(item, data)])
        assert len(res) == 1
        item_id, success, err = res[0]
        assert success is False
        assert "Validation failed: Missing 'RNA' channel" in err

        updated_item = manager["item2:0"]
        assert updated_item.processed is False
        assert "Validation failed: Missing 'RNA' channel" in updated_item.error
