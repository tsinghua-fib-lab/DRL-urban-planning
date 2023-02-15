from typing import Dict
import numpy as np


def set_land_use_array_from_dict(land_use_array: np.ndarray, land_use_dict: Dict, land_use_id_map: Dict) -> None:
    """Fill the land_use_array with the values in the land_use_dict.

    Args:
      land_use_array: np.ndarray that holds the values of each land use.
      land_use_dict: dict that holds the required values of each land use.
      land_use_id_map: dict that maps the land use names to the land use ids.
    """
    for land_use, value in land_use_dict.items():
        land_use_array[land_use_id_map[land_use]] = value
