import tempfile
import yaml
from pathlib import Path
import pytest
import sqlite3
import json
from microimage.manager import ExperimentManager
from microimage.models import DataItem

def test_annotations():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        data_dir = tmp_path / "data"
        results_dir = tmp_path / "results"
        db_path = tmp_path / "test.db"

        data_dir.mkdir()
        results_dir.mkdir()

        # Let's create an ExperimentManager instance
        manager = ExperimentManager(
            data_dir=data_dir,
            results_dir=results_dir,
            db_path=db_path
        )

        # Add a dummy DataItem manually to the database
        item = DataItem(
            id="test_image.czi:0",
            relative_path=Path("test_image.czi"),
            scene="0",
        )
        item.channels = ["DAPI", "GFP", "RFP"]
        item.size_x = 10
        item.size_y = 10
        item.size_c = 3
        item.scale_x = 1.0
        item.scale_y = 1.0

        # Create the item in DB
        conn = manager._get_connection()
        manager._create_item_in_db(conn, item)
        conn.commit()
        conn.close()

        # 1. Test to_annotation - mapping channels to dictionary
        ann = manager.to_annotation()
        assert "set0" in ann
        assert ann["set0"]["channels"] == {"DAPI": "DAPI", "GFP": "GFP", "RFP": "RFP"}
        assert ann["set0"]["files"] == ["test_image.czi"]

        # 2. Modify annotation and import it
        ann["set0"]["channels"] = {"DAPI": "DAPI_Nuclei", "GFP": "GFP_Signal", "RFP": "RFP"}
        ann["set0"]["metadata"] = {"condition": "treated"}
        
        yaml_path = tmp_path / "annotation.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(ann, f)

        manager.import_annotation(yaml_path)

        # Verify items are updated in DB
        updated_item = manager["test_image.czi:0"]
        assert updated_item.annotated_channels == {"DAPI": "DAPI_Nuclei", "GFP": "GFP_Signal", "RFP": "RFP"}
        assert updated_item.annotated_metadata == {"condition": "treated"}

        # Test to_annotation again, now it should keep the annotated channels
        ann_updated = manager.to_annotation()
        assert ann_updated["set0"]["channels"] == {"DAPI": "DAPI_Nuclei", "GFP": "GFP_Signal", "RFP": "RFP"}

        # 3. Test convert_annotation_to_dataframe
        df = manager.convert_annotation_to_dataframe()
        assert df.loc["test_image.czi:0", "Channel DAPI"] == "DAPI_Nuclei"
        assert df.loc["test_image.czi:0", "Channel GFP"] == "GFP_Signal"
        assert df.loc["test_image.czi:0", "Channel RFP"] == "RFP"
        assert df.loc["test_image.czi:0", "condition"] == "treated"
