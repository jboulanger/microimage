import microimage


exp = microimage.ExperimentManager(
    data_dir="/media/cephfs/sankar/SunTag images/60X Sun images/",
    results_dir="../results",
    db_path="../results/experiment.db",
)
exp.scan_data_directory([".czi", ".lsm"])
