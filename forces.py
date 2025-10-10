import numpy as np

from black_hole import black_hole
from constants import *


def compute_black_hole_accelerations(black_holes: list[black_hole]):
    """
    Calculate accelerations [km/s/s] of black holes

    Inputs:
        black_holes, list of black_hole objects
    """

    for i in range(len(black_holes)):
        black_holes[i].acceleration = np.zeros(
            3
        )  # reset accelerations to prevent accumulation from previous call
        for j in range(len(black_holes)):
            if (
                black_holes[i] == black_holes[j]
            ):  # prevent self-calculation on black hole
                continue
            else:
                # update acceleration at i with new acceleration calculation
                black_holes[i].acceleration += compute_acceleration(
                    black_holes[j].mass,
                    black_holes[i].position,
                    black_holes[j].position,
                )


def compute_acceleration(mass: float, position_1, position_2):
    """
    Compute acceleration [km/s/s] components of a black hole object

    Inputs:
        mass: black hole mass [M_sol]
        position_1, np.array, position [kpc] components (x_1, y_1, z_1) of black hole 1 object
        position_2, np.array, position [kpc] components (x_2, y_2, z_2) of black hole 2 object
    Output:
        acceleration, np.array(), acceleration of black hole 1 due to black hole 2 with components (a_x, a_y, a_z)
    """

    # compute Euclidian distance between position vectors 1 and 2
    distance = compute_distance(position_1, position_2)  # scalar
    # compute position vector between position vectors 1 and 2
    r = compute_position(position_1, position_2)  # vector
    # compute acceleration using a = [(G*M) / (|r|^3)] * r
    acceleration = (G * mass) / (distance**3) * r  # vector

    return acceleration / kpc_to_km  # vector converted to [km/s/s]


def compute_distance(position_1, position_2) -> float:
    """
    Return Euclidian distance [kpc] between position_1 [kpc] and position_2 [kpc]
    using linear algebra in numpy

    Inputs:
        position_1, np.array, position with components (x_1, y_1, z_1)
        position_2, np.array, position with components (x_2, y_2, z_2)
    Output:
        Euclidian distance between position_1 and position_2
    """

    return np.linalg.norm(position_1 - position_2)  # Euclidian distance calculation


def compute_position(position_1, position_2):
    """
    Compute the position [kpc] vector between position_1 [kpc] and position_2 [kpc]

    Inputs:
        position_1, np.array, position with components (x_1, y_1, z_1)
        position_2, np.array, position with components (x_2, y_2, z_2)
    Output:
        r, np.array, position vector between position_1 and position_2
    """

    # compute position vector through vectorization
    return position_2 - position_1
