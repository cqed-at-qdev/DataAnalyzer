from dataanalyzer.data_handling import save_load, update_version


dataset_0_0_2 = save_load.load_dataset("dataset_0_0_2.nc")


dataset_0_0_3 = update_version.update_to_latest(dataset_0_0_2)

dataset_0_0_2_again = update_version.update_to_version(dataset_0_0_3, "0.0.2")


dataset_0_0_2_again = update_version.update_to_version(dataset_0_0_2, "0.0.1")