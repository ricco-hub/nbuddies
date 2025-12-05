from tqdm import tqdm
from .Forces import *
from .mergers import find_merger_new
from .merge_gravitree import build_tree
import os
import pickle
import numpy as np
import copy

KM_PER_KPC = 3.0856776e16 # number of km in kpc for using velocity to update position

def load_data_pkl(filename, path = None):
    ''' 
    Load the input position or velocity from a pickle file
    
    Inputs: 
    filename - string of the filename, to be given by the IC team
    path - string of the common directory, given by the Git(?)

    Output:
    Array of the loaded data having shape (N,3),
    where N is number of particles, 3 is the coordinates
    or
    list of Blackhole class objects having shape (batch_size, N) along with the metadata
    '''
    if (path is None):
        file_path = filename
    else:
        file_path = os.path.join(path, filename)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    '''
    checking if the loaded file is the dictionary of the data of blackholes with metadata, 
    or a simple pickle file with information given by IC team
    '''
    if isinstance(data, dict):
        if "data" in data:
            print("Loaded pickle file with metadata.")
            return data["data"], data          #returns both the data and the metadata
        else:
            print("Loaded pickle file without metadata.")
            return data                        #returns the data only - useful for IC team
    else: 
        raise ValueError(f"Expected a .pkl file, got: {os.path.basename(file_path)}")

def save_data_pkl(files, filename, path):
    ''' 
    Save the updated position or velocity arrays as pickle files
    
    Inputs:
    filename - string of the filename, chosen by the timesteps team
    path - string of the common directory, given by the Git(?)
    data - list of Blackhole class objects having shape (batch_size, N) 

    Output:
    None
    '''
    # check if the folder exit
    os.makedirs(path) if not os.path.exists(path) else None 
    # join folder with filename
    file_path = os.path.join(path, filename)

    data, time, delta_t, tot_time, num_steps = files # unpack the input tuple
    if data is None:
        print("No time-evolved data to save")   #if used by the IC team
    with open(file_path, 'wb') as f:
        pickle.dump({
            "time units" : "s",
            "distance units" : "kpc",
            "velocity units" : "km/s",
            "number of particles" : len(data[0]), # number of particles in each batch
            "total time" : tot_time,
            "number of steps per batch" : num_steps,
            "delta_t" : delta_t,
            "time" : time,
            "data" : data
        }, f)

def update_params(data, tot_time, num_steps, delta_t, path, leapfrog, use_tree, use_dynamic_criterion, ALPHA, THETA_0):
    ''' 
    Carries out the integration for each particle for one time step
    
    Inputs:
    data - list of BH objects
    delta_t - float value of the time step for evolution
    tot_time - float value of the total time for evolution
    num_steps - integer value of the number of time steps to be saved in each batch
                given by batch = tot_time // (num_steps * delta_t)
    path - string of the common directory, given by the Git(?)
    leapfrog - True if want leapfrog integration, False if want simple Euler integration (default True)
    use_dynamic_criterion : bool
        determines if dynamic or geometric criterion is used for node approximation
    ALPHA : float
        accuracy parameter for dynamic criterion
    THETA_0 : float
        accuracy parameter for geometric criterion

    Output:
    None - All output files are saved as picke file
    '''

    batch_idx = 0
    count = 0 # goes from 0 to num_steps - 1, used to check when to save the data  
    data_lst = [data] # initialized with the starting data, stores the evolved data batch-wise
    for i, timestep in enumerate(np.arange(0, tot_time, delta_t)): # for each time step, carry out the evolution for all BHs
        if (leapfrog):
            # Leapfrog Integration
            result = leapfrog_integrator(data, delta_t, timestep, use_tree, use_dynamic_criterion, ALPHA, THETA_0)
        else:
            # Euler integration
            result = euler_integrator(data, delta_t, use_tree, use_dynamic_criterion, ALPHA, THETA_0)
        count += 1
        # store the updated result after each time step in data_list
        data_lst.append(copy.deepcopy(result))
        if count == num_steps:
            files = [data_lst, np.arange(count)*delta_t, delta_t, tot_time, num_steps]
            save_data_pkl(files, f'data_batch{batch_idx}.pkl', path)  # saving as a pkl file right now
            batch_idx += 1
            # resets the values for the next batch
            count = 0
            data_lst = []
    
    # Save any remaining timesteps
    if (data_lst):
        files = [data_lst, delta_t, tot_time, num_steps]
        save_data_pkl(files, f"data_batch{batch_idx}.pkl", path)


def update_params_adaptive_timestep(data, tot_time, num_steps, eta, path, leapfrog, use_tree, use_dynamic_criterion, ALPHA, THETA_0):
    ''' 
    Carries out the integration for each particle for one time step
    
    Inputs:
    data - list of BH objects
    eta - constant that decides the relation between timestep and acceleration, jerk, snap. Needs to be optimized.  
    tot_time - float value of the total time for evolution
    num_steps - integer value of the number of time steps to be saved in each batch
                given by batch = tot_time // (num_batches * delta_t)
    path - string of the common directory, given by the Git(?)
    leapfrog - True if want leapfrog integration, False if want simple Euler integration (default True)
    use_dynamic_criterion : bool
        determines if dynamic or geometric criterion is used for node approximation
    ALPHA : float
        accuracy parameter for dynamic criterion
    THETA_0 : float
        accuracy parameter for geometric criterion

    Output:
    None - All output files are saved as picke file
    '''


    batch_idx = 0
    count = 0 # goes from 0 to num_steps - 1, used to check when to save the data  
    data_lst = [data] # initialized with the starting data, stores the evolved data batch-wise
    running_time = 0   # time elapsed in the simulation, will end when running_time == tot_time
    times = np.zeros(num_steps)
    delta_t = 0 
    # needs to be initialized with units because recalculate_acceleration now assigns units acceleration
    #Removed ureg.s from running_time, times, and delta_t


    recalculate_dynamics(data, use_tree, use_dynamic_criterion, ALPHA, THETA_0) # Get acceleration with current position

    with tqdm(total=tot_time, desc="Simulation Progress") as pbar:
        while running_time < tot_time:
            # block to decide the delta_t value for this iteration - 
            delta_t_BH = np.zeros(len(data)) #Removed ureg.s 
            for i, BH in enumerate(data):
                delta_t_BH[i] = comp_adaptive_dt(BH.acceleration, BH.jerk, BH.snap, eta, tot_time)  # compute adaptive value
            delta_t = np.min(delta_t_BH)   # choose the minimum among all BHs

            pbar.update(delta_t)

             # Convert delta_t to Myr for display (1 Myr = 3.15576e13 seconds)
            delta_t_myr = delta_t / 3.15576e13
            pbar.set_postfix({"time_elapsed": f"{running_time / tot_time * 100:.2f}%", "Δt": f"{delta_t_myr:.2e} Myr"})
            #Removed pint associated .magnitude
            # print(f"time_elapsed: {running_time.magnitude/tot_time*100:.2f}% of tot_time; delta_t for this iteration: {delta_t.to('Myr')}")
            # this statement can be useful to keep a check on how fast the simulation is proceeding for a given eta

            if (leapfrog):
                # Leapfrog Integration
                result = leapfrog_integrator(data, delta_t, running_time, use_tree, use_dynamic_criterion, ALPHA, THETA_0)
            else:
                # Euler integration
                result = euler_integrator(data, delta_t, use_tree, use_dynamic_criterion, ALPHA, THETA_0)
            
            if running_time != 0: # skip the first timestep (?)
                tree, radius = build_tree(result)
                consumed_ids = set()
                new_result = []

                for bh in result:
                    if id(bh) in consumed_ids:
                        continue

                    merged_bh, consumed_list = find_merger_new(bh, tree, radius)

                    # mark all consumed BHs by ID
                    for c in consumed_list:
                        consumed_ids.add(id(c))

                    # Add merged BH to next timestep
                    new_result.append(merged_bh)

                result = new_result
            
            running_time += delta_t
            times[count] = running_time
            count += 1
            data_lst.append(copy.deepcopy(result))
            if count == num_steps:
                files = [data_lst, times, delta_t, tot_time, num_steps] 
                # the above way of saving means we are saving the value of timestep in the last simulation of each batch
                # will keep it like that for now so that we can access the typical values of timesteps, later can just save the eta as metadata

                save_data_pkl(files, f'data_batch{batch_idx}.pkl', path)  # saving as a pkl file right now
                batch_idx += 1
                # resets the values for the next batch
                count = 0
                data_lst = []
            
            #recalculate_dynamics(data, use_tree, use_dynamic_criterion, ALPHA, THETA_0) 
            # these need to be done before the next computation of dt (next iteration of the loop)

    # Save any remaining timesteps
    if (data_lst):
        files = [data_lst, times[:count], delta_t, tot_time, num_steps]
        save_data_pkl(files, f"data_batch{batch_idx}.pkl", path)



def leapfrog_integrator(data, delta_t, timestep, use_tree, use_dynamic_criterion : bool, ALPHA, THETA_0):
    """
    Updating position and velocity of BH objects with conserving phase space volume (symplectic integrator).
    
    1. First Kick: Update velocity by half step using current acceleration
    2. Drift: Update position by full step using updated velocity
    3. Recalculate acceleration with new positions
    4. Second Kick: Update velocity by another half step using new acceleration
    
    Inputs:
    data - list of BH objects
    delta_t - The timestep for each round of update
    time_step - current time step index (to check if acceleration needs recalculation)
    use_dynamic_criterion : bool
        determines if dynamic or geometric criterion is used for node approximation
    ALPHA : float
        accuracy parameter for dynamic criterion
    THETA_0 : float
        accuracy parameter for geometric criterion

    Output:
    result - list of Blackhole object
    
    """
    delta_half = delta_t / 2
    if timestep == 0:
        recalculate_dynamics(data, use_tree, use_dynamic_criterion, ALPHA, THETA_0) # Get acceleartion with current position

    result = []

    # First Kick and Drift
    for BH in data:
        BH.velocity += BH.acceleration * delta_half # Update the velocity with half of timestep
        BH.position +=  (BH.velocity/ KM_PER_KPC) * delta_t  # Update the position with the new velocity and with full timestep
    # need to update velocity and position without units in order to keep it compatible with the rest of the code
    # Removed pint associated .magnitude

    # Recalculation of the acceleration
    recalculate_dynamics(data, use_tree, use_dynamic_criterion, ALPHA, THETA_0)

    # Last Kick
    for BH in data:
        BH.velocity += BH.acceleration * delta_half# Update the velocity with half of timestep and updated acceleration
        result.append(BH.copy())
         # Removed pint associated .magnitude
    return result
            
def euler_integrator(data, delta_t, use_tree, use_dynamic_criterion, ALPHA, THETA_0):
    """
    Euler integration of the position and velocity according to
    r(t+Delta t) = r(t) + v(t) * Delta t
    v(t+Delta t) = v(t) + a(t) * Delta t

    Inputs:
    data - list of BH objects
    delta_t - The timestep for each round of update
    use_dynamic_criterion : bool
        determines if dynamic or geometric criterion is used for node approximation
    ALPHA : float
        accuracy parameter for dynamic criterion
    THETA_0 : float
        accuracy parameter for geometric criterion
    
    Output:
    result - list of Blackhole object
    
    """
    recalculate_dynamics(data, use_tree, use_dynamic_criterion, ALPHA, THETA_0)  # provided in Forces.py
    result = []
    for BH in data:  # assumes the BH objects are already loaded with initial values
        BH.position +=  (BH.velocity/ KM_PER_KPC) * delta_t  # Euler integration (formula given above)
        BH.velocity += BH.acceleration * delta_t # Euler integration (formula given above)
        # Removed pint associated .magnitude
        result.append(BH.copy())
    return result
 
#Function to compute adaptive timestep
def comp_adaptive_dt(acc, jerk, snap, eta, tot_time):
    """
    Inputs: acc, jerk, snap, and eta 
    We will need the magnitude for acc, jerk, and snap
    Using our function from class:
    dt = eta * ((jerk/acc)**2 + (snap/acc))**(-1/2)
    Because we are dividing by acc we need to ensure that we don't divide by zero
    We can divide instead by a_mag_safe = np.maximum(a_mag, "small factor")
    Returns the adaptive timestep
    """
    #Calculates magnitudes for acc, jerk, and snap
    a_mag = np.linalg.norm(acc)
    j_mag = np.linalg.norm(jerk)
    s_mag = np.linalg.norm(snap)

    adaptive_factor = np.sqrt((j_mag / a_mag)**2 + (s_mag / a_mag)) 
    if adaptive_factor < 1 / (tot_time / 1000):  # prevent extremely large timesteps
        dt = (tot_time / 1000)  #in seconds but removed ureg.s
    else:
        dt = eta / adaptive_factor #computes dt 

    # print("a_mag = ", a_mag)
    # print("j_mag = ", j_mag)
    # print("s_mag = ", s_mag)  # these will have proper units attached with them
    # print('dt =', dt) # this will have the correct units (seconds) as well

    return dt


def simulation(initial_file, output_folder, tot_time, nsteps, delta_t, adaptive_dt, eta, leapfrog, 
               use_tree, use_dynamic_criterion, ALPHA, THETA_0):
    """
    Wrapper Function for the simulation of time evolve N-body Problem
    
    Inputs:
    initial_file : Path name to the file contained the initial condition file
    output_folder : Path to folder for the save time steps
    tot_time : total amount of time of the simulation
    delta_t : size of the timestep
    nsteps : number of steps for each saving of the batch
    adaptive_dt: whether to use the adaptive timestep formula using eta
    eta: the constant for the adpative timestep formula. Cannot be none if adpative_dt is True. 
    leapfrog: Integration technique. Default to Leap frog, False uses Euler
    use_tree : whether to use the BHT code for force calculation (default True)
    use_dynamic_criterion : bool
        determines if dynamic or geometric criterion is used for node approximation
    ALPHA : float
        accuracy parameter for dynamic criterion
    THETA_0 : float
        accuracy parameter for geometric criterion
    
    Outputs:
        
    """

    # load initial condition
    data, inital = load_data_pkl(initial_file) # should be a list of BH objects

    # Run Simulation
    if adaptive_dt:
        if eta is None:
            raise ValueError("Adaptive timestepping (adaptive_dt = True) requires a value of eta to be given.")
        else:
            update_params_adaptive_timestep(data, tot_time, nsteps, eta, output_folder, leapfrog, use_tree, use_dynamic_criterion, ALPHA, THETA_0)
    else:
        update_params(data, tot_time, nsteps, delta_t, output_folder, leapfrog, use_tree, use_dynamic_criterion, ALPHA, THETA_0)


# print('\nNeed to call the simulation function properly to ensure it works though :)')

# Example usage by calling the simulation function using arbitrary parameters and names
# simulation1 = simulation('initial_conditions.pkl', './', 100, 0.01, 10)