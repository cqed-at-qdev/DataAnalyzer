# Author: Malthe Asmus Marciniak Nielsen
from typing import Tuple

from dataanalyzer.utilities.valueclass import Valueclass
from dataanalyzer.utilities.data_conversion import labber2valueclass, json2valueclass


def load_labber_file(labber_path: str, insepct: bool = False) -> Tuple[Valueclass, ...]:
    return labber2valueclass(labber_path, insepct=insepct)


def load_json_file(
    json_path: str, insepct: bool = False
) -> Tuple[list[Valueclass], list[Valueclass]]:
    return json2valueclass(json_path, insepct=insepct)


if __name__ == "__main__":
    fpath = r"C:\Users\T5_2\Desktop\quantum machines demo\data20230105\152337_state_after_protective_freq_vs_theta.json"
    load_json_file(fpath, insepct=True)
