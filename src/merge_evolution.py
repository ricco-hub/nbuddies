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
    '''
    if (path is None):
        file_path = filename
    else:
        file_path = os.path.join(path, filename)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        if "data" in data:
            print("Loaded pickle file with metadata.")
            return data["data"], data
        else:
            print("Loaded pickle file without metadata.")
            return data
    else:
        raise ValueError(f"Expected a .pkl file, got: {os.path.basename(file_path)}")


def save_data_pkl(files, filename, path):
    '''
    Save data. Accepts either a dict with keys (data, time, dt, total_time, num_steps)
    or a sequence [data, time, dt, total_time, num_steps] for backward compatibility.
    '''
    os.makedirs(path) if not os.path.exists(path) else None
    file_path = os.path.join(path, filename)

    # Normalize to dict
    if isinstance(files, dict):
        data = files.get("data")
        time = files.get("time")
        delta_t = files.get("dt") if "dt" in files else files.get("delta_t")
        tot_time = files.get("total_time") if "total_time" in files else files.get("tot_time")
        num_steps = files.get("num_steps")
    else:
        # expected sequence
        try:
            data, time, delta_t, tot_time, num_steps = files
        except Exception:
            raise ValueError("files must be either dict or sequence of 5 items")

    if data is None:
        print("No time-evolved data to save")

    # Determine number of particles for metadata (if possible)
    try:
        n_particles = len(data[0])
    except Exception:
        # fallback when data empty
        n_particles = 0

    with open(file_path, 'wb') as f:
        pickle.dump({
            "time units" : "s",
            "distance units" : "kpc",
            "velocity units" : "km/s",
            "number of particles" : n_particles,
            "total time" : tot_time,
            "number of steps per batch" : num_steps,
            "delta_t" : delta_t,
            "time" : time,
            "data" : data
        }, f)


# ------------------ Non-adaptive (fixed dt) integrator loop ------------------
def update_params(data, tot_time, num_steps, delta_t, path, leapfrog, use_tree, use_dynamic_criterion, ALPHA, THETA_0):
    batch_idx = 0
    count = 0

    # current state (list of BH objects)
    state = copy.deepcopy(data)

    # accumulate snapshots for batch
    data_lst = [copy.deepcopy(state)]
    times = [0.0]

    t = 0.0
    while t < tot_time:
        if leapfrog:
            result = leapfrog_integrator(state, delta_t, t, use_tree, use_dynamic_criterion, ALPHA, THETA_0)
        else:
            result = euler_integrator(state, delta_t, use_tree, use_dynamic_criterion, ALPHA, THETA_0)

        t += delta_t
        count += 1

        # update state to result and append
        state = result
        data_lst.append(copy.deepcopy(state))
        times.append(t)

        if count == num_steps:
            files = {
                "data": data_lst,
                "time": times,
                "dt": delta_t,
                "total_time": tot_time,
                "num_steps": len(data_lst)
            }
            save_data_pkl(files, f'data_batch{batch_idx}.pkl', path)
            batch_idx += 1
            # prepare for next batch: start from last state
            count = 0
            data_lst = [copy.deepcopy(state)]
            times = [t]

    # Save any remaining timesteps
    if data_lst:
        files = {
            "data": data_lst,
            "time": times,
            "dt": delta_t,
            "total_time": tot_time,
            "num_steps": len(data_lst)
        }
        save_data_pkl(files, f"data_batch{batch_idx}.pkl", path)


# ------------------ Adaptive-timestep integrator loop ------------------
def update_params_adaptive_timestep(data, tot_time, num_steps, eta, path, leapfrog, use_tree, 
                                   use_dynamic_criterion, ALPHA, THETA_0, merge_delay_steps=10):
    """
    Added parameter:
    merge_delay_steps : int, default 10
        Number of integration steps to complete before enabling mergers
    """
    batch_idx = 0
    count = 0
    
    integration_step = 0  # counts actual integration steps completed

    state = copy.deepcopy(data)

    running_time = 0.0
    time_lst = [running_time]
    data_lst = [copy.deepcopy(state)]  # Save initial state (step 0)
    result = data

    # ensure initial accelerations/jerks/snaps computed
    recalculate_dynamics(state, use_tree, use_dynamic_criterion, ALPHA, THETA_0)

    print(f"Starting simulation with merge delay of {merge_delay_steps} steps")

    with tqdm(total=tot_time, desc="Simulation Progress") as pbar:
        while running_time < tot_time:
            # compute per-BH suggested dt
            delta_t_BH = np.zeros(len(state))
            for i, BH in enumerate(state):
                delta_t_BH[i] = comp_adaptive_dt(BH.acceleration, BH.jerk, BH.snap, eta, tot_time)

            # choose minimum and ensure it's finite
            delta_t = np.min(delta_t_BH)
            if not np.isfinite(delta_t) or delta_t <= 0:
                delta_t = tot_time / 1000.0

            pbar.update(delta_t)

            delta_t_myr = delta_t / 3.15576e13
            pbar.set_postfix({
                "time_elapsed": f"{running_time / tot_time * 100:.2f}%", 
                "Δt": f"{delta_t_myr:.2e} Myr",
                "step": integration_step,
                "N": len(state)
            })

            # integrate one step
            if leapfrog:
                result = leapfrog_integrator(state, delta_t, running_time, use_tree, use_dynamic_criterion, ALPHA, THETA_0)
            else:
                result = euler_integrator(state, delta_t, use_tree, use_dynamic_criterion, ALPHA, THETA_0)

            N_before = len(result)

            # Increment step counter BEFORE merge check
            integration_step += 1

            # perform merging only after delay period
            if integration_step > merge_delay_steps:  # Use > instead of >=
                if integration_step == merge_delay_steps + 1:
                    print(f"\n🔀 Merging enabled at step {integration_step}")
                
                N_before = len(result)
                tree, radius = build_tree(result)
                consumed_ids = set()
                new_result = []
                merges_done = 0
                max_merges_per_step = 1                
                for bh in result:
                    if id(bh) in consumed_ids:
                        continue
                    merged_bh, consumed_list = find_merger_new(bh, tree, radius)
                    if len(consumed_list) > 1 and merges_done < max_merges_per_step:
                        merges_done += 1
                        for c in consumed_list:
                            consumed_ids.add(id(c))
                        new_result.append(merged_bh)
                        if merges_done >= max_merges_per_step:
                            for other in result:
                                if id(other) not in consumed_ids:
                                    new_result.append(other)
                            break
                    else:
                        if all(id(c) not in consumed_ids for c in consumed_list):
                            new_result.append(merged_bh)
            
                if merges_done == 0:
                    existing_ids = {id(bh) for bh in new_result}
                    for bh in result:
                        if id(bh) not in existing_ids and id(bh) not in consumed_ids:
                            new_result.append(bh)

                result = new_result
                
            N_after = len(result)
            if N_before != N_after:
                print(f"  Merge at step {integration_step}: {N_before} → {N_after} BHs")

            # advance time
            running_time += delta_t
            count += 1

            # update state to result and append
            state = result
            data_lst.append(copy.deepcopy(state))
            time_lst.append(running_time)

            if count == num_steps:
                files = {
                    "data": data_lst,
                    "time": time_lst,
                    "dt": delta_t,
                    "total_time": tot_time,
                    "num_steps": len(data_lst)
                }
                save_data_pkl(files, f'data_batch{batch_idx}.pkl', path)
                batch_idx += 1
                count = 0
                data_lst = [copy.deepcopy(state)]
                time_lst = [running_time]

    # Save any remaining timesteps
    if data_lst:
        files = {
            "data": data_lst,
            "time": time_lst,
            "dt": delta_t,
            "total_time": tot_time,
            "num_steps": len(data_lst)
        }
        save_data_pkl(files, f"data_batch{batch_idx}.pkl", path)
    
    print(f"\nSimulation complete. Final N = {len(state)}")


# ------------------ Integrators ------------------
def leapfrog_integrator(data, delta_t, timestep, use_tree, use_dynamic_criterion : bool, ALPHA, THETA_0):
    delta_half = delta_t / 2.0
    if timestep == 0:
        recalculate_dynamics(data, use_tree, use_dynamic_criterion, ALPHA, THETA_0)

    # First kick & drift
    for BH in data:
        BH.velocity = BH.velocity + BH.acceleration * delta_half
        BH.position = BH.position + (BH.velocity / KM_PER_KPC) * delta_t

    # recompute dynamics
    recalculate_dynamics(data, use_tree, use_dynamic_criterion, ALPHA, THETA_0)

    # Last kick and return copies
    result = []
    for BH in data:
        BH.velocity = BH.velocity + BH.acceleration * delta_half
        result.append(BH.copy())
    return result


def euler_integrator(data, delta_t, use_tree, use_dynamic_criterion, ALPHA, THETA_0):
    recalculate_dynamics(data, use_tree, use_dynamic_criterion, ALPHA, THETA_0)
    result = []
    for BH in data:
        BH.position = BH.position + (BH.velocity / KM_PER_KPC) * delta_t
        BH.velocity = BH.velocity + BH.acceleration * delta_t
        result.append(BH.copy())
    return result


# ------------------ Adaptive timestep helper ------------------
def comp_adaptive_dt(acc, jerk, snap, eta, tot_time):
    a_mag = np.linalg.norm(acc)
    j_mag = np.linalg.norm(jerk)
    s_mag = np.linalg.norm(snap)

    # prevent divide-by-zero
    a_safe = max(a_mag, 1e-30)

    adaptive_factor = (j_mag / a_safe)**2 + (s_mag / a_safe)
    # sqrt only if positive
    if adaptive_factor <= 0 or not np.isfinite(adaptive_factor):
        return tot_time / 1000.0
    adaptive_factor = np.sqrt(adaptive_factor)

    # prevent extremely small adaptive_factor
    if adaptive_factor < 1e-20:
        return tot_time / 1000.0

    dt = eta / adaptive_factor

    # clamp
    if dt > tot_time / 1000.0:
        dt = tot_time / 1000.0

    return dt


def simulation(initial_file, output_folder, tot_time, nsteps, delta_t, adaptive_dt, eta, leapfrog, 
               use_tree, use_dynamic_criterion, ALPHA, THETA_0):
    # load initial condition
    data, inital = load_data_pkl(initial_file)

    if adaptive_dt:
        if eta is None:
            raise ValueError("Adaptive timestepping (adaptive_dt = True) requires a value of eta to be given.")
        else:
            update_params_adaptive_timestep(data, tot_time, nsteps, eta, output_folder, leapfrog, use_tree, use_dynamic_criterion, ALPHA, THETA_0)
    else:
        update_params(data, tot_time, nsteps, delta_t, output_folder, leapfrog, use_tree, use_dynamic_criterion, ALPHA, THETA_0)
