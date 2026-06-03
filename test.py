import os
import shutil
from pathlib import Path
from microimage import Reader, ExperimentManager
from pylibCZIrw import czi as pyczi
import dask.array as da


filename = "/media/cephfs/sankar/Images for Jerome/SunTag images/111125_GFP 488_PCNT 647_SunBicD2 FL CCmut new_1.czi"
filename = "/home/jeromeb/Downloads/w96_A1+A2.czi"
# parse the file
with Reader(filename) as f:
    for k, s in enumerate(f.iter_scenes()):
        print(k, s.shape)


def test_manager_refactor():
    """Basic test to verify the 'data' and 'results' table refactoring."""
    test_dir = Path("./test_experiment")
    data_dir = test_dir / "data"
    results_dir = test_dir / "results"
    db_path = results_dir / "test.db"

    # Cleanup and setup
    if test_dir.exists():
        shutil.rmtree(test_dir)
    data_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)

    # Create manager
    exp = ExperimentManager(data_dir=data_dir, results_dir=results_dir, db_path=db_path)

    print(f"Database initialized at: {db_path}")

    # Check if tables were created with new names
    conn = exp._get_connection()
    tables = [
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    print(f"Tables found: {tables}")
    assert "data" in tables
    assert "results" in tables
    conn.close()


test_manager_refactor()
