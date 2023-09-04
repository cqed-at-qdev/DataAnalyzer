import numpy as np


def check_if_valid(qubit_freq, flux_bias, dc_bias):
    # Check if input settings are valid
    if [qubit_freq, flux_bias, dc_bias].count(None) != 2:
        raise ValueError("Exactly one of qubit_freq, flux_bias, dc_bias must be defined")


# Transmon Spectrum Conversions ------------------------------------------------


def flux_bias_to_freq(flux_bias, max_freq=1, Ec=1, d=0):
    """Function to convert flux bias to qubit frequency.

    Args:
        flux_bias (array): Input data (in V).
        max_freq (float): Maximum frequency
        Ec (float): Charging energy

    Returns:
        array: qubit frequency
    """

    # d = 0  # (0<d<1 for asymmetric transmons)
    return (max_freq + Ec) * (d**2 + (1 - d**2) * (np.cos(np.pi * flux_bias)) ** 2) ** 0.25 - Ec


def freq_to_flux_bias(freq, max_freq=1, Ec=1, d=0):
    """Function to convert qubit frequency to flux bias.

    Args:
        freq (array): Input data (in V).
        max_freq (float): Maximum frequency
        Ec (float): Charging energy

    Returns:
        array: flux bias
    """

    # d = 0  # (0<d<1 for asymmetric transmons)
    return np.arccos(((((freq + Ec) / (max_freq + Ec)) ** 4 - d**2) * (1 / (1 - d**2))) ** 0.5) / np.pi


def dc_bias_to_flux_bias(dc_bias, v_per_phi0=1, flux_offset=0):
    """Function to convert dc bias to flux bias.

    Args:
        dc_bias (array): Input data (in V).
        v_per_phi0 (float): Period of frequency spectrum (in V)
        flux_offset (float): Flux offset (in units of Phi_0)

    Returns:
        array: flux bias
    """

    return dc_bias / v_per_phi0 + flux_offset


def flux_bias_to_dc_bias(flux_bias, v_per_phi0=1, flux_offset=0):
    """Function to convert flux bias to dc bias.

    Args:
        flux_bias (array): Input data (in V).
        v_per_phi0 (float): Period of frequency spectrum (in V)
        flux_offset (float): Flux offset (in units of Phi_0)

    Returns:
        array: dc bias
    """

    return (flux_bias - flux_offset) * v_per_phi0


def transmon_spectrum_conversion(
    max_freq, Ec, d, v_per_phi0, flux_offset, qubit_frequency=None, flux_bias=None, dc_bias=None
):
    # check_if_valid(qubit_frequency, flux_bias, dc_bias)

    if qubit_frequency is not None:
        flux_bias = freq_to_flux_bias(qubit_frequency, max_freq, Ec, d)
        dc_bias = flux_bias_to_dc_bias(flux_bias, v_per_phi0, flux_offset)

    elif flux_bias is not None:
        qubit_frequency = flux_bias_to_freq(flux_bias, max_freq, Ec, d)
        dc_bias = flux_bias_to_dc_bias(flux_bias, v_per_phi0, flux_offset)

    else:
        flux_bias = dc_bias_to_flux_bias(dc_bias, v_per_phi0, flux_offset)
        qubit_frequency = flux_bias_to_freq(flux_bias, max_freq, Ec, d)

    return qubit_frequency, flux_bias, dc_bias
