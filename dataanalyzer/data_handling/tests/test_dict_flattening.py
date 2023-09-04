from dataanalyzer.data_handling import save_load, update_version
import xarray as xr
import uncertainties.unumpy as unp
from uncertainties import ufloat as ufloat

dataset_0_0_2 = save_load.load_dataset("dataset_0_0_2.nc")
dataset_0_0_3 = update_version.update_to_latest(dataset_0_0_2)

dataset_0_0_3.attrs["test_dict"] = {"test": 2}
dataset_0_0_3.readout__final__I__avg.attrs["test_dict"] = {"test": 123, "nested": {"testing": 1123}}

save_load.save_dataset(dataset_0_0_3, "dataset_0_0_3.nc")
dataset_0_0_3_loaded = save_load.load_dataset("dataset_0_0_3.nc")


print(dataset_0_0_3.equals(dataset_0_0_3_loaded))
# This fails due to equality in unumpy, i.e. ufloat(1, 1) == ufloat(1, 1) is False


# Comparing in version 0.0.2 confirms that the dataset is saved correctly
dataset_0_0_2_loaded = update_version.update_to_version(dataset_0_0_3_loaded, "0.0.2")
dataset_0_0_2_again = update_version.update_to_version(dataset_0_0_3, "0.0.2")

print(dataset_0_0_2_again.equals(dataset_0_0_2_loaded))
