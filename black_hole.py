import numpy as np


class black_hole:
    def __init__(self, mass: float, position: list[float], velocity: list[float]):
        """
        Instantiate black hole class with initial mass, position, and velocity

        Inputs:
            mass, initial black hole mass [M_sol]
            position, initial position [kpc] with components (x, y, z)
            velocity, initial velocity [km/s] with components (v_x, v_y, v_z)
        """

        self.mass = mass
        self.position = np.array(position)  # casting to np.array for vectorization
        self.velocity = np.array(velocity)  # casting to np.array for vectorization
        self.acceleration = np.zeros(3)  # np.array of initial acceleration [km/s/s]
        # with components (a_x, a_y, a_z) = (0, 0, 0) initially

    def __eq__(self, other) -> bool:
        """
        Check if two black holes are the same by comparing masses,
        and position and velocity components

        Input:
            other, black_hole, second black hole to check against
        Output:
            True if black holes are equal; False if not
        """

        # check black hole masses
        if other.mass != self.mass:
            return False
        for i in range(3):
            # check position components
            if other.position[i] != self.position[i]:
                return False
            # check velocity components
            if other.velocity[i] != self.velocity[i]:
                return False

        return True
