#N-Body Data Structure
import numpy as np

class BlackHole():
    def __init__(self, mass: float, position: list[float], velocity: list[float],
                acceleration : list[float] = [0,0,0], 
                 jerk: list[float] = [0, 0, 0], 
                 snap: list[float] = [0, 0, 0]):
        """
        Define data structure --> struct 
        mass, position (3D), velocity (3D), and acceleration (initialized to zero by default)
        Validate data before assigning them

        Parameters
        ----------
        mass : float
            mass of black hole
        position : list[float]
            position vector of black hole in kpc
        velocity : list[float])
            velocity vector of black hole in km/s
        acceleration : list[float], default [0,0,0]
            acceleration vector of black hole in km/s^2
        jerk : list[float], default is [0,0,0]
            3D jerk vector (in km/s^3)
        snap : list[float], default is [0,0,0]
            3D snap vector (in km/s^4)
        """
        assert mass > 0, "Mass must be positive" #checks mass is positive 
        assert len(position) == 3, "Position must be a 3D vector" #checks that position is a vector
        assert len(velocity) == 3, "Velocity must be a 3D vector" #checks that velocity is a vector
        assert len(acceleration) == 3, "Acceleration must be a 3D vector"
        assert len(jerk) == 3, "Jerk must be a 3D vector"
        assert len(snap) == 3, "Snap must be a 3D vector"
       
        self.mass = mass 
        self.position = np.array(position) 
        self.velocity = np.array(velocity) 
        self.acceleration = np.array(acceleration)
        self.jerk = np.array(jerk)
        self.snap = np.array(snap)

    def displacement(self, other):
        """
        Computes displacement vector from self to another black hole

        Parameters
        ----------
        other : BlackHole
            The black hole to which displacement is being calculated

        Returns
        -------
        list[float]
            displacement vector between other and self, points towards other.
        """
        return other.position - self.position 

    def __eq__(self, other): 
        """
        Checks if two blacks holes are the same by checking all except acceleration.

        Parameters
        ----------
        other : BlackHole
            The black hole being compared to

        Returns
        -------
        bool
            Whether or not these black holes are the same.
        """
        if other.mass != self.mass: 
            return False
        for i in range(3):
            if other.position[i] != self.position[i]: 
                return False
            if other.velocity[i] != self.velocity[i]: 
                return False 
        return True 

    def copy(self):
        """
        Returns copy of the black hole object

        Returns
        -------
        BlackHole
            Copy of the black hole object
        """
        return BlackHole(self.mass, self.position, self.velocity, self.acceleration, self.jerk, self.snap)
