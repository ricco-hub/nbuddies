"""
N-Body Black Hole Position Predictor

Usage:
    python train_nbody.py path/to/your/dataset.pkl

The dataset should be a pickle file containing:
    - 'ICs': Initial conditions (positions, velocities)
    - 'Final_Data': Final positions and velocities
    - 'Ns': Number of particles per simulation
    - 'Ms': Total mass per simulation
    - 'Rs': Scale radius per simulation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle as pkl
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import sys


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def pad_bh_array(arr, target_N):
    """Pads array with zeros to reach target_N rows."""
    pad = target_N - arr.shape[0]
    if pad > 0:
        arr = np.vstack([arr, np.zeros((pad, arr.shape[1]))])
    return arr


def find_valid_indices(dataset):
    """
    Find indices of valid simulations.
    
    A simulation is invalid if:
    - Final number of particles > Initial number of particles
    - Any other data corruption
    
    Returns:
        valid_indices: list of valid simulation indices
        skipped_indices: list of skipped simulation indices with reasons
    """
    valid_indices = []
    skipped_indices = []
    
    for idx in range(len(dataset['ICs'])):
        try:
            # Get initial conditions
            ic_data = dataset['ICs'][idx]
            ic_snapshot = ic_data['data'] if isinstance(ic_data, dict) else ic_data
            n_initial = len(ic_snapshot)
            
            # Get final conditions
            final_snapshot = dataset['Final_Data'][idx]['data']
            n_final = len(final_snapshot)
            
            # Check if final > initial (invalid)
            if n_final > n_initial:
                skipped_indices.append((idx, f"Final particles ({n_final}) > Initial particles ({n_initial})"))
                continue
            
            # Check if counts match expected N
            expected_N = int(dataset['Ns'][idx])
            if n_initial != expected_N:
                skipped_indices.append((idx, f"Initial particles ({n_initial}) != Expected N ({expected_N})"))
                continue
            
            # Valid simulation
            valid_indices.append(idx)
            
        except Exception as e:
            skipped_indices.append((idx, f"Error reading data: {str(e)}"))
            continue
    
    return valid_indices, skipped_indices


def generate_3d_plots(model, data_loader, dataset, indices, output_folder, set_name):
    """
    Generate 3D plots comparing predicted vs actual positions.
    
    Args:
        model: Trained model
        data_loader: DataLoader for the data
        dataset: Original dataset (for getting N, M, R values)
        indices: List of simulation indices
        output_folder: Where to save plots
        set_name: "train" or "test" (for plot titles)
    
    Returns:
        all_errors: List of mean errors for each simulation
    """
    os.makedirs(output_folder, exist_ok=True)
    
    model.eval()
    all_errors = []
    
    for batch_idx, (X, y, mask) in enumerate(data_loader):
        with torch.no_grad():
            pred = model(X, mask)
        
        for i in range(X.shape[0]):
            sim_num = batch_idx * data_loader.batch_size + i
            
            # Make sure we don't go out of bounds
            if sim_num >= len(indices):
                break
                
            actual_sim_idx = indices[sim_num]
            n_particles = int(mask[i].sum())
            
            pred_disp = pred[i, :n_particles, :3].numpy()
            actual_disp = y[i, :n_particles, :3].numpy()
            init_pos = X[i, :n_particles, :3].numpy()
            
            pred_final = init_pos + pred_disp
            actual_final = init_pos + actual_disp
            
            error = np.linalg.norm(pred_final - actual_final, axis=1).mean()
            all_errors.append(error)
            
            # Get simulation parameters
            N = int(dataset['Ns'][actual_sim_idx])
            M = dataset['Ms'][actual_sim_idx]
            R = dataset['Rs'][actual_sim_idx]
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.scatter(actual_final[:, 0], actual_final[:, 1], actual_final[:, 2], 
                       c='blue', s=50, alpha=0.6, label='Actual')
            ax.scatter(pred_final[:, 0], pred_final[:, 1], pred_final[:, 2], 
                       c='orange', s=50, alpha=0.6, label='Predicted')
            
            # Draw error lines
            for j in range(n_particles):
                ax.plot([actual_final[j, 0], pred_final[j, 0]],
                        [actual_final[j, 1], pred_final[j, 1]],
                        [actual_final[j, 2], pred_final[j, 2]],
                        'gray', alpha=0.3, linewidth=1)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{set_name.upper()} Sim {sim_num + 1} | N={N} | M={M:.2e} | R={R:.2f} | Error={error:.3f}')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(f'{output_folder}/{set_name}_sim_{sim_num:03d}.png')
            plt.close()
    
    return all_errors


# =============================================================================
# DATASET CLASS
# =============================================================================

class NBodyDataset(Dataset):
    """
    Dataset for N-body simulations.
    
    Takes initial conditions and returns normalized inputs (X) and 
    displacement targets (y).
    """
    
    def __init__(self, dataset, valid_indices, stats=None):
        """
        Args:
            dataset: The full dataset dictionary
            valid_indices: List of indices to include
            stats: Normalization stats (if None, computed from data)
        """
        self.valid_indices = valid_indices
        
        self.ICs = dataset["ICs"]
        self.final = dataset["Final_Data"]
        self.Ns = dataset["Ns"]
        self.Ms = dataset["Ms"]
        self.Rs = dataset["Rs"]
        self.max_N = max(int(dataset["Ns"][i]) for i in self.valid_indices)
        
        if stats is None:
            self._compute_stats()
        else:
            self.pos_std = stats['pos_std']
            self.vel_std = stats['vel_std']
            self.logM_mean = stats['logM_mean']
            self.logM_std = stats['logM_std']
            self.N_mean = stats['N_mean']
            self.N_std = stats['N_std']
            self.R_mean = stats['R_mean']
            self.R_std = stats['R_std']
        
    def _compute_stats(self):
        """Compute normalization statistics from the data."""
        all_pos, all_vel = [], []
        
        for idx in self.valid_indices:
            ic_data = self.ICs[idx]
            ic_snapshot = ic_data['data'] if isinstance(ic_data, dict) else ic_data
            final_snapshot = self.final[idx]['data']
            
            all_pos.extend([bh.position for bh in ic_snapshot])
            all_pos.extend([bh.position for bh in final_snapshot])
            all_vel.extend([bh.velocity for bh in ic_snapshot])
            all_vel.extend([bh.velocity for bh in final_snapshot])
        
        all_pos = np.array(all_pos)
        all_vel = np.array(all_vel)
        
        self.pos_std = all_pos.std()
        self.vel_std = all_vel.std()
        
        log_M = np.log10([self.Ms[i] for i in self.valid_indices])
        self.logM_mean, self.logM_std = log_M.mean(), log_M.std()
        
        Ns = [self.Ns[i] for i in self.valid_indices]
        Rs = [self.Rs[i] for i in self.valid_indices]
        self.N_mean, self.N_std = np.mean(Ns), np.std(Ns)
        self.R_mean, self.R_std = np.mean(Rs), np.std(Rs)
        
        print(f"  Position std: {self.pos_std:.2f}")
        print(f"  Velocity std: {self.vel_std:.2f}")
        print(f"  Log10(M) range: {log_M.min():.1f} to {log_M.max():.1f}")
    
    def get_stats(self):
        """Return normalization stats for use by test set."""
        return {
            'pos_std': self.pos_std,
            'vel_std': self.vel_std,
            'logM_mean': self.logM_mean,
            'logM_std': self.logM_std,
            'N_mean': self.N_mean,
            'N_std': self.N_std,
            'R_mean': self.R_mean,
            'R_std': self.R_std
        }

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        
        N = int(self.Ns[actual_idx])
        M = float(self.Ms[actual_idx])
        R = float(self.Rs[actual_idx])

        ic_data = self.ICs[actual_idx]
        ic_snapshot = ic_data['data'] if isinstance(ic_data, dict) else ic_data
        final_snapshot = self.final[actual_idx]['data']

        IC_pos = np.array([bh.position for bh in ic_snapshot], dtype=np.float32)
        IC_vel = np.array([bh.velocity for bh in ic_snapshot], dtype=np.float32)
        final_pos = np.array([bh.position for bh in final_snapshot], dtype=np.float32)
        final_vel = np.array([bh.velocity for bh in final_snapshot], dtype=np.float32)

        # Normalize
        IC_pos_norm = IC_pos / self.pos_std
        IC_vel_norm = IC_vel / self.vel_std
        final_pos_norm = final_pos / self.pos_std
        final_vel_norm = final_vel / self.vel_std

        # Pad to max_N
        IC_pos_padded = pad_bh_array(IC_pos_norm, self.max_N)
        IC_vel_padded = pad_bh_array(IC_vel_norm, self.max_N)
        final_pos_padded = pad_bh_array(final_pos_norm, self.max_N)
        final_vel_padded = pad_bh_array(final_vel_norm, self.max_N)

        # Mask: 1 for real particles, 0 for padding
        mask = np.zeros(self.max_N, dtype=np.float32)
        mask[:N] = 1.0

        # Normalize parameters
        params = np.array([
            (N - self.N_mean) / self.N_std,
            (np.log10(M) - self.logM_mean) / self.logM_std,
            (R - self.R_mean) / self.R_std
        ], dtype=np.float32)
        params_tiled = np.tile(params, (self.max_N, 1))

        # Input: position, velocity, params
        X = np.concatenate([IC_pos_padded, IC_vel_padded, params_tiled], axis=1)
        
        # Output: displacement (how much each particle moved)
        displacement_pos = final_pos_padded - IC_pos_padded
        displacement_vel = final_vel_padded - IC_vel_padded
        y = np.concatenate([displacement_pos, displacement_vel], axis=1)

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32)
        )


# =============================================================================
# MODEL
# =============================================================================

class BHPredictor(nn.Module):
    """
    Neural network that predicts particle displacements.
    
    Each particle gets information about:
    - Its own position and velocity (6 numbers)
    - Simulation parameters N, M, R (3 numbers)
    - Average position of other particles (3 numbers)
    
    Total input: 12 numbers per particle
    Output: 6 numbers (position and velocity displacement)
    """
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, x, mask):
        # Get positions (first 3 features)
        positions = x[:, :, :3]
        
        # Compute average position of OTHER particles
        mask_expanded = mask.unsqueeze(-1)
        sum_of_positions = (positions * mask_expanded).sum(dim=1, keepdim=True)
        num_real = mask.sum(dim=1, keepdim=True).unsqueeze(-1)
        
        sum_of_others = sum_of_positions - positions
        avg_of_others = sum_of_others / (num_real - 1 + 0.0001)
        
        # Combine original features with neighbor info
        x_with_neighbors = torch.cat([x, avg_of_others], dim=-1)
        
        return self.net(x_with_neighbors)


# =============================================================================
# LOSS FUNCTION
# =============================================================================

def masked_mse_loss(pred, target, mask):
    """
    Mean squared error that ignores padded particles.
    """
    mask_expanded = mask.unsqueeze(-1)
    squared_error = (pred - target) ** 2
    masked_error = squared_error * mask_expanded
    return masked_error.sum() / (mask_expanded.sum() * pred.shape[-1] + 1e-8)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main(data_path):
    """
    Main training and evaluation function.
    
    Args:
        data_path: Path to the pickle file containing simulation data
    """
    
    # Create output directories
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/train_predictions", exist_ok=True)
    os.makedirs(f"{output_dir}/test_predictions", exist_ok=True)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    
    print(f"\nLoading data from: {data_path}")
    with open(data_path, "rb") as f:
        dataset = pkl.load(f)
    
    print(f"\nDataset overview:")
    print(f"  Total simulations: {len(dataset['ICs'])}")
    print(f"  N range: {min(dataset['Ns']):.0f} to {max(dataset['Ns']):.0f}")
    print(f"  Mass range: {min(dataset['Ms']):.2e} to {max(dataset['Ms']):.2e}")
    print(f"  Radius range: {min(dataset['Rs']):.2f} to {max(dataset['Rs']):.2f}")
    
    # =========================================================================
    # FIND VALID SIMULATIONS
    # =========================================================================
    
    print(f"\nChecking for invalid simulations...")
    valid_indices, skipped_indices = find_valid_indices(dataset)
    
    print(f"  Valid simulations: {len(valid_indices)}")
    print(f"  Skipped simulations: {len(skipped_indices)}")
    
    if skipped_indices:
        print(f"\n  Skipped details:")
        for idx, reason in skipped_indices:
            print(f"    Simulation {idx}: {reason}")
    
    if len(valid_indices) < 5:
        print("\nError: Not enough valid simulations to train (need at least 5)")
        sys.exit(1)
    
    # =========================================================================
    # SPLIT INTO TRAIN AND TEST
    # =========================================================================
    
    np.random.seed(42)
    shuffled_indices = valid_indices.copy()
    np.random.shuffle(shuffled_indices)
    
    split_point = int(0.8 * len(shuffled_indices))
    train_indices = shuffled_indices[:split_point]
    test_indices = shuffled_indices[split_point:]
    
    print(f"\nData split:")
    print(f"  Training simulations: {len(train_indices)}")
    print(f"  Test simulations: {len(test_indices)}")
    
    # Create datasets
    print(f"\nComputing normalization statistics...")
    train_dataset = NBodyDataset(dataset, train_indices)
    train_stats = train_dataset.get_stats()
    
    test_dataset = NBodyDataset(dataset, test_indices, stats=train_stats)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # For generating plots, we need non-shuffled loaders
    train_loader_no_shuffle = DataLoader(train_dataset, batch_size=4, shuffle=False)
    
    # =========================================================================
    # TRAIN
    # =========================================================================
    
    print("\n" + "="*50)
    print("TRAINING")
    print("="*50)
    
    model = BHPredictor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    train_losses = []
    test_losses = []
    
    # Early stopping
    best_test_loss = float('inf')
    best_model_state = None
    patience = 30
    patience_counter = 0
    
    for epoch in range(300):
        # Training
        model.train()
        total_train_loss = 0
        for X, y, mask in train_loader:
            optimizer.zero_grad()
            pred = model(X, mask)
            loss = masked_mse_loss(pred, y, mask)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Testing
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for X, y, mask in test_loader:
                pred = model(X, mask)
                loss = masked_mse_loss(pred, y, mask)
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # Early stopping
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with test loss: {best_test_loss:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), f"{output_dir}/trained_model.pt")
    print(f"Model saved to {output_dir}/trained_model.pt")
    
    # =========================================================================
    # PLOT 1: TRAIN VS TEST LOSS
    # =========================================================================
    
    print("\n" + "="*50)
    print("GENERATING PLOTS")
    print("="*50)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test (unseen data)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(f'{output_dir}/train_vs_test_loss.png')
    plt.close()
    print(f"Saved: {output_dir}/train_vs_test_loss.png")
    
    # =========================================================================
    # PLOT 2: PREDICTED VS ACTUAL DISPLACEMENT (BOTH SETS)
    # =========================================================================
    
    model.eval()
    
    # Collect data for both sets
    def collect_displacement_data(loader):
        pred_mags = []
        actual_mags = []
        with torch.no_grad():
            for X, y, mask in loader:
                pred = model(X, mask)
                for i in range(X.shape[0]):
                    n_particles = int(mask[i].sum())
                    pred_disp = pred[i, :n_particles, :3].numpy()
                    actual_disp = y[i, :n_particles, :3].numpy()
                    pred_mags.extend(np.linalg.norm(pred_disp, axis=1))
                    actual_mags.extend(np.linalg.norm(actual_disp, axis=1))
        return np.array(pred_mags), np.array(actual_mags)
    
    train_pred_mag, train_actual_mag = collect_displacement_data(train_loader_no_shuffle)
    test_pred_mag, test_actual_mag = collect_displacement_data(test_loader)
    
    # Plot both
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].scatter(train_actual_mag, train_pred_mag, alpha=0.3, s=10)
    axes[0].plot([0, train_actual_mag.max()], [0, train_actual_mag.max()], 'r--', label='Perfect prediction')
    axes[0].set_xlabel('Actual displacement magnitude')
    axes[0].set_ylabel('Predicted displacement magnitude')
    axes[0].set_title('Training Set: Predicted vs Actual Movement')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].scatter(test_actual_mag, test_pred_mag, alpha=0.3, s=10)
    axes[1].plot([0, test_actual_mag.max()], [0, test_actual_mag.max()], 'r--', label='Perfect prediction')
    axes[1].set_xlabel('Actual displacement magnitude')
    axes[1].set_ylabel('Predicted displacement magnitude')
    axes[1].set_title('Test Set: Predicted vs Actual Movement')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/displacement_scatter_both.png')
    plt.close()
    print(f"Saved: {output_dir}/displacement_scatter_both.png")
    
    # =========================================================================
    # PLOT 3: ERROR HISTOGRAM (BOTH SETS)
    # =========================================================================
    
    train_errors = np.abs(train_pred_mag - train_actual_mag)
    test_errors = np.abs(test_pred_mag - test_actual_mag)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(train_errors, bins=50, edgecolor='black')
    axes[0].set_xlabel('Displacement Prediction Error')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Training Set Error Distribution\nMean: {train_errors.mean():.3f}, Std: {train_errors.std():.3f}')
    
    axes[1].hist(test_errors, bins=50, edgecolor='black')
    axes[1].set_xlabel('Displacement Prediction Error')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Test Set Error Distribution\nMean: {test_errors.mean():.3f}, Std: {test_errors.std():.3f}')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_histogram_both.png')
    plt.close()
    print(f"Saved: {output_dir}/error_histogram_both.png")
    
    # =========================================================================
    # PLOT 4: 3D COMPARISONS FOR TRAINING SET
    # =========================================================================
    
    print(f"\nGenerating 3D plots for training set...")
    train_3d_errors = generate_3d_plots(
        model, 
        train_loader_no_shuffle, 
        dataset, 
        train_indices, 
        f'{output_dir}/train_predictions',
        'train'
    )
    print(f"Saved: {output_dir}/train_predictions/train_sim_XXX.png")
    
    # =========================================================================
    # PLOT 5: 3D COMPARISONS FOR TEST SET
    # =========================================================================
    
    print(f"Generating 3D plots for test set...")
    test_3d_errors = generate_3d_plots(
        model, 
        test_loader, 
        dataset, 
        test_indices, 
        f'{output_dir}/test_predictions',
        'test'
    )
    print(f"Saved: {output_dir}/test_predictions/test_sim_XXX.png")
    
    # =========================================================================
    # PLOT 6: ERROR VS SIMULATION PARAMETERS (BOTH SETS)
    # =========================================================================
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Training set
    train_Ns = [dataset['Ns'][idx] for idx in train_indices]
    train_Ms = [dataset['Ms'][idx] for idx in train_indices]
    train_Rs = [dataset['Rs'][idx] for idx in train_indices]
    
    axes[0, 0].scatter(train_Ns, train_3d_errors, alpha=0.7)
    axes[0, 0].set_xlabel('N (number of particles)')
    axes[0, 0].set_ylabel('Mean Error')
    axes[0, 0].set_title('Training: Error vs N')
    
    axes[0, 1].scatter(np.log10(train_Ms), train_3d_errors, alpha=0.7)
    axes[0, 1].set_xlabel('log10(M)')
    axes[0, 1].set_ylabel('Mean Error')
    axes[0, 1].set_title('Training: Error vs Mass')
    
    axes[0, 2].scatter(train_Rs, train_3d_errors, alpha=0.7)
    axes[0, 2].set_xlabel('R (scale radius)')
    axes[0, 2].set_ylabel('Mean Error')
    axes[0, 2].set_title('Training: Error vs Radius')
    
    # Test set
    test_Ns = [dataset['Ns'][idx] for idx in test_indices]
    test_Ms = [dataset['Ms'][idx] for idx in test_indices]
    test_Rs = [dataset['Rs'][idx] for idx in test_indices]
    
    axes[1, 0].scatter(test_Ns, test_3d_errors, alpha=0.7)
    axes[1, 0].set_xlabel('N (number of particles)')
    axes[1, 0].set_ylabel('Mean Error')
    axes[1, 0].set_title('Test: Error vs N')
    
    axes[1, 1].scatter(np.log10(test_Ms), test_3d_errors, alpha=0.7)
    axes[1, 1].set_xlabel('log10(M)')
    axes[1, 1].set_ylabel('Mean Error')
    axes[1, 1].set_title('Test: Error vs Mass')
    
    axes[1, 2].scatter(test_Rs, test_3d_errors, alpha=0.7)
    axes[1, 2].set_xlabel('R (scale radius)')
    axes[1, 2].set_ylabel('Mean Error')
    axes[1, 2].set_title('Test: Error vs Radius')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_vs_parameters_both.png')
    plt.close()
    print(f"Saved: {output_dir}/error_vs_parameters_both.png")
    
    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Dataset:")
    print(f"  Total simulations: {len(dataset['ICs'])}")
    print(f"  Valid simulations: {len(valid_indices)}")
    print(f"  Skipped: {len(skipped_indices)}")
    
    print(f"\nTraining set results ({len(train_indices)} simulations):")
    print(f"  Mean error: {np.mean(train_3d_errors):.3f}")
    print(f"  Std error:  {np.std(train_3d_errors):.3f}")
    print(f"  Best:       {np.min(train_3d_errors):.3f}")
    print(f"  Worst:      {np.max(train_3d_errors):.3f}")
    
    print(f"\nTest set results ({len(test_indices)} simulations):")
    print(f"  Mean error: {np.mean(test_3d_errors):.3f}")
    print(f"  Std error:  {np.std(test_3d_errors):.3f}")
    print(f"  Best:       {np.min(test_3d_errors):.3f}")
    print(f"  Worst:      {np.max(test_3d_errors):.3f}")
    
    print(f"\nAll results saved to: {output_dir}/")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_nbody.py path/to/dataset.pkl")
        print("\nExample:")
        print("  python train_nbody.py training_data/dataset2_100sims_HA.pkl")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    if not os.path.exists(data_path):
        print(f"Error: File not found: {data_path}")
        sys.exit(1)
    
    main(data_path)