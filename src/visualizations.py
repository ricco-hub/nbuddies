import os
import shutil
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


nbuddies_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def movie_3D(sim_name : str, tail_length: int = 10, tot_nstep_eta = None):
    """
    Loads data and makes movie of motion in 3D space with tails behind them

    Parameters
    ----------
    sim_name : str
        The name of the simulation run
    tail_length : int, default 10
        length of tail trailing behind points
    tot_nstep_eta: str, used to dynamically save the resulting movies with info about 
                    total time (sec), num of timesteps per batch, eta for adaptive timestep computation
    """
    
    #set up
    # Creating output folder for animation frames if it doesn't exist
    if os.path.exists(nbuddies_path+"/movie_dump/"+sim_name): # check if dir exists
        shutil.rmtree(nbuddies_path+"/movie_dump/"+sim_name) #pruge old images  
    os.makedirs(nbuddies_path+"/movie_dump/"+sim_name) # create dir path
    
    #getting info from sim end
    last_batch_num = _find_last_batch_num(sim_name) # find number corresponding to last data batch number
    
    # Load last batch and initialize data structures
    with open(nbuddies_path + '/data/' + sim_name + f"/data_batch{last_batch_num}.pkl", 'rb') as file: # open dir of batches
        data = pickle.load(file)['data'][0] # load pk data files with last batch number
    N = len(data) # length of data files

    #Calculate the maximum plot range based on particle positions
    max_range = 0
    max_mass = 0.0
    for n in range(N):
        max_mass = max(max_mass, data[n].mass)
        if np.linalg.norm(data[n].position) > max_range: # if Euclidian distance is greater than max_range
            max_range = np.linalg.norm(data[n].position) # set max_range to Euclidian distance
    max_range *= 2 # add buffer by increasing max_range by 200%

    #getting info from sim start
    with open(nbuddies_path + '/data/'+ sim_name + "/data_batch0.pkl", 'rb') as file:
        init_data = pickle.load(file)['data'][0]
    #Create 3D array to store tail positions
    plotting_data = np.zeros([N, 3, tail_length]) # instantiate array of zeros with dimensions N x 3 x tail_length
    min_mass = np.inf
    for n in range(N):
        min_mass = min(min_mass, init_data[n].mass)
        for t in range(tail_length):
            plotting_data[n, :, t] = init_data[n].position # append positions to plotting_data
    
    #init mass array
    masses = np.zeros(N)

    #set up cmap
    viridis_dark = colors.LinearSegmentedColormap.from_list('viridis_dark', 
                                                plt.cm.viridis(np.linspace(0, 0.7, 256))).reversed()

    #generating movie frames
    #Loop through batch to make frames
    for i in range(last_batch_num + 1):
        #slide data window forward
        for j in range(tail_length - 1):
            plotting_data[:,:,j] = plotting_data[:,:,j+1] # move data window forward by 1
        #Load current frame data
        with open(nbuddies_path + '/data/' + sim_name + f"/data_batch{i}.pkl", 'rb') as file: # open dir to pk files
            file = pickle.load(file)
            data = file['data'][0] # load pk files
            time = file['time'][0]

        positions = np.array([obj.position for obj in data])
        masses    = np.array([obj.mass for obj in data])

        #plot
        fig = plt.figure() # instantiate figure
        ax = fig.add_subplot(111, projection='3d') # create 3D subplot

        #plotting points
        points = ax.scatter(positions[:,0], positions[:,1], positions[:,2], c=masses, s=100/len(data), marker="o", 
                            vmin=min_mass, vmax=max_mass, cmap = viridis_dark, alpha=1) # plot positions

        #assign colorbar
        cbar = plt.colorbar(points, ax=ax, pad=0.1)
        cbar.set_label(r"Mass $M_{\odot}$")

        # Set plot labels and limits
        # create x,y,z labels + title
        ax.set_xlabel('X [kpc]')
        ax.set_ylabel('Y [kpc]')
        ax.set_zlabel('Z [kpc]')
        ax.set_title(fr'$N={N}$ Black Hole Trajectories')

        # set x,y,z figure limits
        ax.set_xlim( - max_range/2, max_range/2)
        ax.set_ylim( - max_range/2, max_range/2)
        ax.set_zlim( - max_range/2, max_range/2)

         #Convert time from seconds to Myr (1 Myr = 3.15576e13 seconds) without pint
        time_myr = time / 3.15576e13
        ax.set_title(f"t={time_myr:.3f} Myr")
        
        plt.tight_layout()
        #Save current frame as png
        plt.savefig(nbuddies_path + "/movie_dump/"+sim_name+f"/trajectories_{i}.png", dpi=300, bbox_inches='tight') # save fig in dir
        plt.close()

    _recompile_movie_3D(sim_name, tot_nstep_eta) # Combine saved frames into video using ffmpeg

def movie_3D_new(sim_name: str, tail_length: int = 10, tot_nstep_eta=None):
    """
    Loads data and makes movie of motion in 3D space with tails behind them.
    Now properly iterates through *all snapshots* in each batch rather than
    always using snapshot 0.
    """

    # Create output folder for animation frames
    if os.path.exists(nbuddies_path + "/movie_dump/" + sim_name):
        shutil.rmtree(nbuddies_path + "/movie_dump/" + sim_name)
    os.makedirs(nbuddies_path + "/movie_dump/" + sim_name)

    # --- Load last batch to determine sizes ---
    last_batch_num = _find_last_batch_num(sim_name)

    # Load last batch to get final N, max_range, max_mass
    with open(nbuddies_path + '/data/' + sim_name + f"/data_batch{last_batch_num}.pkl", 'rb') as file:
        final_batch = pickle.load(file)["data"]
        final_snapshot = final_batch[-1]       # last snapshot in last batch

    N_final = len(final_snapshot)

    # Compute plotting ranges
    max_range = 0
    max_mass = 0
    for bh in final_snapshot:
        max_range = max(max_range, np.linalg.norm(bh.position))
        max_mass = max(max_mass, bh.mass)
    max_range *= 2.0

    # --- Load initial batch to set initial tail + min_mass ---
    with open(nbuddies_path + '/data/' + sim_name + "/data_batch0.pkl", 'rb') as file:
        init_batch = pickle.load(file)["data"][0]

    N_init = len(init_batch)
    min_mass = min(bh.mass for bh in init_batch)

    # Initialize tail
    plotting_data = np.zeros((N_init, 3, tail_length))
    for n in range(N_init):
        for t in range(tail_length):
            plotting_data[n, :, t] = init_batch[n].position

    # colormap
    viridis_dark = colors.LinearSegmentedColormap.from_list(
        'viridis_dark', plt.cm.viridis(np.linspace(0, 0.7, 256))
    ).reversed()

    # --------- MAIN LOOP: iterate through all batches and snapshots ----------
    frame_id = 0   # unique frame counter

    for batch_i in range(last_batch_num + 1):

        # load entire batch
        with open(nbuddies_path + '/data/' + sim_name + f"/data_batch{batch_i}.pkl", 'rb') as file:
            file = pickle.load(file)
            batch_data = file["data"]    # list of snapshots
            batch_times = file["time"]   # list of times

        # iterate through every snapshot in this batch
        for snap_i in range(len(batch_data)):
            snapshot = batch_data[snap_i]
            time = batch_times[snap_i]

            positions = np.array([obj.position for obj in snapshot])
            masses    = np.array([obj.mass for obj in snapshot])

            # shift tail window
            for j in range(tail_length - 1):
                plotting_data[:, :, j] = plotting_data[:, :, j+1]

            # update newest tail point
            for n in range(len(snapshot)):
                plotting_data[n, :, -1] = snapshot[n].position

            # --- Plotting ---
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Scatter plot
            points = ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                c=masses, s=100 / len(snapshot),
                vmin=min_mass, vmax=max_mass,
                cmap=viridis_dark, marker="o", alpha=1
            )

            # colorbar
            cbar = plt.colorbar(points, ax=ax, pad=0.1)
            cbar.set_label(r"Mass $M_{\odot}$")

            # axes
            ax.set_xlabel('X [kpc]')
            ax.set_ylabel('Y [kpc]')
            ax.set_zlabel('Z [kpc]')

            ax.set_xlim(-max_range/2, max_range/2)
            ax.set_ylim(-max_range/2, max_range/2)
            ax.set_zlim(-max_range/2, max_range/2)

            # convert time to Myr
            time_myr = time / 3.15576e13
            ax.set_title(f"t = {time_myr:.3f} Myr")

            plt.tight_layout()

            # Save frame
            out_path = (
                nbuddies_path +
                f"/movie_dump/{sim_name}/trajectories_{frame_id}.png"
            )
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()

            frame_id += 1

    # Reassemble movie
    _recompile_movie_3D(sim_name, tot_nstep_eta)

def _recompile_movie_3D(sim_name, tot_nstep_eta):
    """
    Deletes movie if it exists then recreates it by compiling the pngs in movie_dump

    Parameters
    ----------
    sim_nam : str
        name of simulation
    tot_nstep_eta: str
        used to dynamically save the resulting movies with info about total time (sec), num of timesteps per batch, eta for adaptive timestep computation
    """
    visuals_dir = os.path.join(nbuddies_path, "visuals", sim_name)
    movie_dump_dir = os.path.join(nbuddies_path, "movie_dump", sim_name)

    # normalize BEFORE building patterns
    visuals_dir = visuals_dir.replace("\\", "/")
    movie_dump_dir = movie_dump_dir.replace("\\", "/") 

    if not os.path.exists(visuals_dir):
        os.makedirs(visuals_dir)

    outfile = f"{visuals_dir}/trajectories_{tot_nstep_eta}.mkv"
    if os.path.exists(outfile):
        os.remove(outfile)

    # THIS MUST MATCH ACTUAL FILENAME NUMBERING
    input_pattern = f"{movie_dump_dir}/trajectories_%d.png"

    cmd = (
        f'ffmpeg -y -framerate 12 '
        f'-i "{input_pattern}" '
        f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" '
        f'-c:v libx264 -pix_fmt yuv420p '
        f'"{outfile}"'
    )
    os.system(cmd)

def _find_last_batch_num(sim_name) -> int:
    """
    finds num of last batch file saved

    Parameters
    ----------
    sim_name : str
        The name of the simulation
    Returns
    -------
    int
        num of last batch file saved
    """

    i = 0
    while os.path.exists(nbuddies_path + "/data/" + sim_name + f"/data_batch{i}.pkl"): # while path of ith data batch exists
        i += 1 # increment i
    return i - 1 # i is number corresponding to last data batch number


def radial_position_plot(sim_name):
    last_batch_num = _find_last_batch_num(sim_name)

    #getting info from sim start
    with open(nbuddies_path + '/data/'+ sim_name + "/data_batch0.pkl", 'rb') as file:
        file = pickle.load(file)
        init_data = file['data']
    
    n_batch = len(file['time'])
    print(n_batch)
    N = len(init_data[0])

    r_points = np.zeros([N, last_batch_num*n_batch])
    t_points = np.zeros(last_batch_num*n_batch)

    masses = np.zeros(N)

    for n in range(N):
        masses[n] = init_data[0][n].mass

    for i in range(last_batch_num):
        with open(nbuddies_path + '/data/'+ sim_name + f"/data_batch{i}.pkl", 'rb') as file:
            file = pickle.load(file)
        for j in range(n_batch):
            k = i*n_batch + j
            for n in range(N):
                r_points[n,k] = np.linalg.norm(file["data"][j][n].position)
            # Convert time from seconds to Myr (1 Myr = 3.15576e13 seconds) without using pint
            t_points[k] = file["time"][j] / 3.15576e13
    
    #set up cmap
    viridis_dark = colors.LinearSegmentedColormap.from_list('viridis_dark', plt.cm.viridis(np.linspace(0, 0.7, 256))).reversed()

    norm = colors.Normalize(vmin=np.min(masses), vmax=np.max(masses))

    fig = plt.figure()
    ax = fig.add_subplot()

    line_colors = viridis_dark(norm(masses))

    for n in range(N):
        ax.plot(t_points, r_points[n], color=line_colors[n])

    ax.set_xlabel("t (Myr)")
    ax.set_ylabel("r (kpc)")
    ax.set_yscale('log')
    ax.set_title("Radial Position over Time")

    sm = plt.cm.ScalarMappable(cmap=viridis_dark, norm=norm)
    sm.set_array([])  # This line is needed for colorbar to work
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label(r"Mass $M_{\odot}$")

    plt.tight_layout()
    plt.savefig(nbuddies_path + "/visuals/" + sim_name + "/radial_positions.png")


def radial_position_plot_new(sim_name):
    last_batch_num = _find_last_batch_num(sim_name)

    #getting info from sim start
    with open(nbuddies_path + '/data/'+ sim_name + "/data_batch0.pkl", 'rb') as file:
        file = pickle.load(file)
        init_data = file['data']
    
    n_batch = len(file['time'])
    print(n_batch)
    N = len(init_data[0])

    total_steps = last_batch_num * n_batch
    r_points = np.full((N, total_steps), np.nan)
    t_points = np.zeros(total_steps)

    masses = np.zeros(N)

    for n in range(N):
        masses[n] = init_data[0][n].mass

    for i in range(last_batch_num):
        with open(nbuddies_path + '/data/'+ sim_name + f"/data_batch{i}.pkl", 'rb') as file:
            file = pickle.load(file)

        for j in range(n_batch):
            # N = len(file["data"][j]) # this is changing (somehow)
            k = i*n_batch + j

            snapshot = file["data"][j]
            N_current = len(snapshot)
            for n in range(N_current):
                r_points[n,k] = np.linalg.norm(snapshot[n].position)

            t_points[k] = file["time"][j] / 3.15576e13

            # for n in range(N):
            #     # print("len of file data:", len(file["data"][j])) # this is changing!
            #     r_points[n,k] = np.linalg.norm(file["data"][j][n].position)
            # t_points[k] = file["time"][j].to('Myr').magnitude
    
    #set up cmap
    viridis_dark = colors.LinearSegmentedColormap.from_list('viridis_dark', plt.cm.viridis(np.linspace(0, 0.7, 256))).reversed()

    norm = colors.Normalize(vmin=np.min(masses), vmax=np.max(masses))

    fig = plt.figure()
    ax = fig.add_subplot()

    line_colors = viridis_dark(norm(masses))

    for n in range(N):
        ax.plot(t_points, r_points[n], color=line_colors[n])

    ax.set_xlabel("t (Myr)")
    ax.set_ylabel("r (kpc)")
    ax.set_yscale('log')
    ax.set_title("Radial Position over Time")

    sm = plt.cm.ScalarMappable(cmap=viridis_dark, norm=norm)
    sm.set_array([])  # This line is needed for colorbar to work
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label(r"Mass $M_{\odot}$")

    plt.tight_layout()
    plt.savefig(nbuddies_path + "/visuals/" + sim_name + "/radial_positions.png")

def open_batch_files(sim_name: str):
    last_batch_num = _find_last_batch_num(sim_name)

    for i in range(last_batch_num):
        with open(nbuddies_path + '/data/'+ sim_name + f"/data_batch{i}.pkl", 'rb') as file:
            file = pickle.load(file)
            data = file['data']
        n_batch = len(file['time'])
        N = len(data[0])
        print(f"For data_batch{i}, N={N}")
        # print(f"N={N}")
        # print("N batch: ", n_batch)

def check_lengths(sim_name):
    last_batch_num = _find_last_batch_num(sim_name)
    for i in range(last_batch_num):
        with open(nbuddies_path + '/data/'+ sim_name + f"/data_batch{i}.pkl", 'rb') as file:
            file = pickle.load(file)

        for j, snapshot in enumerate(file["data"]):
            print(f"batch {i}, step {j}, N={len(snapshot)}")



# check_lengths("test_3_mergers_again")
# check_lengths("test_5_mergers"
# open_batch_files("test_5_mergers")
# radial_position_plot_new("test_5_mergers")
# movie_3D_new("test_5_mergers_movie")