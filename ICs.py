import numpy as np
import pickle as pkl
from BlackHoles_Struct import BlackHole

# A function to calcuate the box size based on the distance between two points
def calculate_box_size(distance, n_blackholes):
    box_size = distance * (n_blackholes+1) # length of the cubic box in kpc
    

    return box_size
    

path_to_save_pkl_file = "./"
    
def generate_initial_conditions(n_blackholes, box_size, mass_range = (3, 10**11), vel_scale_kms = 100): # a function to create n_blackholes with masses, positions, and velocities, box_size in kpc (side length of the cubic box)
    #generate initial coniditions for N-body black hole simulation

    # generate masses from a uniform distribution
    masses = np.random.uniform(mass_range[0], mass_range[1], n_blackholes) # Makes n_blackholes random masses, uniformly between the given min and max

    # generate random positions in a cubic box
    # positions are uniformly distributed in [-box_size/2, box_size/2] for each coordinate
    positions = np.random.uniform(-box_size/2, box_size/2 , (n_blackholes, 3)) # (a, b, (n,m)) makes an array of shape (n,m) with random numbers uniformly between a and b. 
    # The center of the box (0,0,0)
    # an example of positions: [[x1,y1,z1],[x2,y2,z2],...,[xn,yn,zn]], where xi,yi,zi are the coordinates of the i-th black hole

    


    # generate velocities from a Maxwell-Boltzmann distribution
    # for each component, use normal distribution with sigma = sqrt(kT/m)
    # since we want temperature in km/s units, we use it directly as velocity scale

    velocities = np.zeros((n_blackholes, 3)) # Prepares an array to hold all velocities (km/s), initially all zeros. Make three random numbers from a bell curve with mean 0 and spread sigma 
    # these are the (vx, vy, vz) for that black hole.

    for i in range(n_blackholes):
        # Maxwell-Boltzmann distribution for each velocity component is normally distributed
        # and scale inversely with sqrt(mass) to account for mass dependence
        sigma = vel_scale_kms / (np.sqrt(masses[i] / np.mean(masses)))  # scale velocity dispersion with mass
        velocities[i] = np.random.normal(0, sigma, 3)  # 3D velocity components # mean velocity = 0 (random normal distribution)

    # create black hole instances
    blackholes = []
    for i in range(n_blackholes):
        bh = BlackHole(masses[i], positions[i], velocities[i])
        blackholes.append(bh)

    

    pkl.dump(blackholes, open(f"{path_to_save_pkl_file}/test1.pkl", "wb")) # save the blackholes list to a pickle file    

    return blackholes, masses, positions, velocities    


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(43)
    
    # Generate initial conditions for 20 black holes
    n = 20 # number of black holes
    distance = 5 # kpc
    box_size = calculate_box_size(distance, n) 
    blackholes, masses, positions, velocities = generate_initial_conditions(n, box_size = box_size)
    
    # Print some statistics
    print(f"Generated {n} black holes")
    print(f"Mass range: {masses.min():.1f} - {masses.max():.1f} M_sun")
    print(f"Position range: [{positions.min():.2f}, {positions.max():.2f}] kpc")
    print(f"Velocity range: [{velocities.min():.2f}, {velocities.max():.2f}] km/s")
    print(f"\nTotal mass: {masses.sum():.1f} M_sun")
    print(f"Center of mass velocity: {np.sum(masses[:, np.newaxis] * velocities, axis=0) / masses.sum()}")
    
    # Example of accessing data through class structure
    
    for i in range(n):
        print(f"\nBlack hole {i+1}:")
        print(f"  Mass: {blackholes[i].mass:.1f} M_sun")
        print(f"  Position: {blackholes[i].position} kpc")
        print(f"  Velocity: {blackholes[i].velocity} km/s")







        
    
        

         