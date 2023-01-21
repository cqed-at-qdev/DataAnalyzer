# Author: Malthe Asmus Marciniak Nielsen

from typing import Tuple

from dataanalyzer.utilities.valueclass import Valueclass
from dataanalyzer.utilities.data_conversion import labber2valueclass, json2valueclass


def load_labber_file(
    labber_path: str, insepct: bool = False
) -> Tuple[list[Valueclass], list[Valueclass]]:
    """Load Labber file and return the measurement parameters and the corresponding results as Valueclass objects.
    
    Args:
        labber_path (str): path to Labber file
        insepct (bool, optional): inspect data. Defaults to False.
        
    Returns:
        Tuple[list[Valueclass], list[Valueclass]]: parameters and results as lists of Valueclass objects.
    """
    return labber2valueclass(labber_path, insepct=insepct)


def load_json_file(
    json_path: str, insepct: bool = False
) -> Tuple[list[Valueclass], list[Valueclass]]:
    """Load JSON file and return the measurement parameters and the corresponding results as Valueclass objects.
    
    Args:
        json_path (str): path to JSON file
        insepct (bool, optional): inspect data. Defaults to False.
        
    Returns:
        Tuple[list[Valueclass], list[Valueclass]]: parameters and results as lists of Valueclass objects.
    """
    return json2valueclass(json_path, insepct=insepct)

