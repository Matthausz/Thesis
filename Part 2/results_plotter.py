import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.lines as mlines

def plot_combined_horizontal(Random_ER_results, Pretrained_ER_results, 
                             Random_3R_results, Pretrained_3R_results, 
                             all_runs, names, labels):
    """
    Plots all results in a horizontally arranged figure with 2 rows and 3 columns.
    
    Each row corresponds to a graph type (3-Regular and ER30) and contains:
    - Pretrained individual runs.
    - Random individual runs.
    - Mean cumulative maximum with confidence intervals.
    """
    
    # Define font sizes
    title_fontsize = 26
    label_fontsize = 22
    tick_fontsize = 20
    legend_fontsize = 22
    subtitle_fontsize = 28

    # Define color scheme
    random_color = '#ff7f0e'      # Orange
    pretrained_color = '#1f77b4'  # Blue

    # Create a figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(24, 16), sharey=False)

    # ------------------------------
    # Helper function to plot individual runs
    # ------------------------------
    def plot_individual_runs(ax, runs, title, color, show_ylabel=True, Title=False, set_legend=False):
        """
        Plots individual runs on the given axis.
        
        Parameters:
        - ax (matplotlib.axes.Axes): The axis to plot on.
        - runs (list of lists): Each inner list contains approximation ratios for a single run.
        - title (str): Title for the subplot.
        - color (str): Color for the individual runs.
        - show_ylabel (bool): Whether to show y-axis labels and ticks.
        - Title (bool): Whether to set the title.
        - set_legend (bool): Whether to set the legend.
        """
        for run in runs:
            iterations = list(range(1, len(run) + 1))
            if len(iterations) < 10:
                ax.plot(iterations, 100 * np.array(run), color='purple', linewidth=3)
            else:
                ax.plot(iterations, 100 * np.array(run), alpha=0.3, color=color)
        
        ax.axhline(y=87.8, color='r', linestyle='--',  linewidth=3,label='GW Threshold')
        if Title:
            ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel("Iteration", fontsize=label_fontsize)
        
        if show_ylabel:
            ax.set_ylabel("Approximation Ratio (%)", fontsize=label_fontsize)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        
        ax.set_ylim(50, 100)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.set_xlim(0, 500)
        ax.grid(True)
        
        # Handle legends to avoid duplicate labels
        if set_legend:
            # Create custom legend handles using Line2D
            pretrained_line = mlines.Line2D([], [], color=pretrained_color, linewidth=2, label='Pretrained QAOA')
            random_line = mlines.Line2D([], [], color=random_color, linewidth=2, label='Randomly initialised QAOA')
            gw_line = mlines.Line2D([], [], color='r', linestyle='--', linewidth=2, label='Goemans-Williamson Threshold')
            
            # Add the legend to the axis
            ax.legend(handles=[pretrained_line, random_line, gw_line],
                      loc='upper left',
                      fontsize=legend_fontsize,
                      framealpha=0.9)

    # ------------------------------
    # Helper function to plot mean with confidence intervals
    # ------------------------------
    def plot_mean_with_ci(ax, runs_group, name, dataset_labels, show_ylabel=True, set_legend=False):
        """
        Plots the mean cumulative maximum with 95% confidence intervals.
        
        Parameters:
        - ax (matplotlib.axes.Axes): The axis to plot on.
        - runs_group (list of lists): Each inner list contains runs for a dataset.
        - name (str): Title for the subplot.
        - dataset_labels (list of str): Labels for each dataset.
        - show_ylabel (bool): Whether to show y-axis labels and ticks.
        - set_legend (bool): Whether to set legend for the subplot.
        """
        for runs, dataset_label in zip(runs_group, dataset_labels):
            # Determine the maximum number of iterations across all runs for this dataset
            max_iterations = max(len(run) for run in runs)

            # Initialize a list to store cumulative maxima for each run
            cumulative_max_runs = []

            for run in runs:
                cumulative_max = []
                current_max = -np.inf
                for val in run:
                    if val > current_max:
                        current_max = val
                    cumulative_max.append(current_max)
                # If the run has fewer iterations than max_iterations, extend with the last max
                if len(cumulative_max) < max_iterations:
                    last_max = cumulative_max[-1]
                    cumulative_max.extend([last_max] * (max_iterations - len(cumulative_max)))
                cumulative_max_runs.append(cumulative_max)

            # Convert to a NumPy array for easier manipulation
            cumulative_max_array = np.array(cumulative_max_runs)  # Shape: (num_runs, max_iterations)

            # Calculate mean and confidence intervals
            mean_values = np.mean(cumulative_max_array, axis=0)
            sem_values = stats.sem(cumulative_max_array, axis=0)
            confidence = 0.95  # 95% confidence interval
            z_score = stats.norm.ppf(1 - (1 - confidence) / 2)  # Two-tailed

            margin_of_error = z_score * sem_values
            lower_bounds = mean_values - margin_of_error
            upper_bounds = mean_values + margin_of_error

            # Define iterations for plotting
            iterations = np.arange(1, max_iterations + 1)

            # Determine color based on dataset label
            if 'Random' in dataset_label:
                color = random_color
            elif 'Pretrained' in dataset_label:
                color = pretrained_color
            else:
                color = 'gray'  # Default color if needed

            print(f"The final mean approximation ratio for {name} using {dataset_label} is {100*mean_values[-1]:.2f}%\n")
            # Plot the mean line
            ax.plot(iterations, 100 * mean_values, label=f'{dataset_label}', linewidth=2, color=color)

            # Plot the confidence interval as a shaded area
            ax.fill_between(iterations, 100 * lower_bounds, 100 * upper_bounds, alpha=0.3, color=color)
        
        # Reference horizontal line
        ax.axhline(y=87.8, color='r', linestyle='--', linewidth=3,label='Goemans-Williamson Threshold')
        
        if show_ylabel:
            ax.set_ylabel("Approximation Ratio (%)", fontsize=label_fontsize)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        
        ax.set_xlabel("Iteration", fontsize=label_fontsize)
        ax.set_ylim(50, 100)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.set_xlim(0, 500)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Handle legends to avoid duplicate labels
        if set_legend:
            handles, labels_ = ax.get_legend_handles_labels()
            unique = dict(zip(labels_, handles))
            ax.legend(unique.values(), unique.keys(), loc='upper left', fontsize=legend_fontsize, framealpha=0.9)

    # ------------------------------
    # Plotting for 3-Regular (Top Row)
    # ------------------------------
    # Pretrained individual runs for 3-Regular (Top-Left)
    plot_individual_runs(
        ax=axes[0, 0],
        runs=Pretrained_3R_results['all_runs'],
        title="",  # No title
        color=pretrained_color,
        show_ylabel=True,    # Show y-axis label and ticks for the first plot in the row
        Title=False,         # Do not set title here
        set_legend=False      # Legend will be set on the middle plot
    )
    
    # Random individual runs for 3-Regular (Top-Middle) - With Title and Legend
    plot_individual_runs(
        ax=axes[0, 1],
        runs=Random_3R_results['all_runs'],
        title="",  # Title will be set as a column title
        color=random_color,
        show_ylabel=False,  
        Title=False,         # No individual subplot title
        set_legend=True      # Set legend on this subplot
    )
    
    # Mean plot for 3-Regular (Top-Right)
    plot_mean_with_ci(
        ax=axes[0, 2],
        runs_group=all_runs[0],
        name=names[1],
        dataset_labels=labels,
        show_ylabel=False,    # y-label shown only on the first plot in the row
        set_legend=False      # Legend moved to the middle plot
    )
    
    # ------------------------------
    # Plotting for ER30 (Bottom Row)
    # ------------------------------
    # Pretrained individual runs for ER30 (Bottom-Left)
    plot_individual_runs(
        ax=axes[1, 0],
        runs=Pretrained_ER_results['all_runs'],
        title="",  # No title
        color=pretrained_color,
        show_ylabel=True,    # Show y-axis label and ticks for the first plot in the row
        Title=False,         # Do not set title here
        set_legend=False      # Legend already set on the top middle plot
    )
    
    # Random individual runs for ER30 (Bottom-Middle) - With Title
    plot_individual_runs(
        ax=axes[1, 1],
        runs=Random_ER_results['all_runs'],
        title="",  # Title will be set as a column title
        color=random_color,
        show_ylabel=False,   
        Title=False,         # No individual subplot title
        set_legend=False      # Legend already set on the top middle plot
    )
    
    # Mean plot for ER30 (Bottom-Right)
    plot_mean_with_ci(
        ax=axes[1, 2],
        runs_group=all_runs[1],
        name=names[0],
        dataset_labels=labels,
        show_ylabel=False,    # y-label shown only on the first plot in the row
        set_legend=False      # Legend already set on the top middle plot
    )

    column_titles = ["(A) Pretrained Runs", "(B) Randomly Initialised Runs", "(C) Mean (Confidence Interval)"]
    for i, title in enumerate(column_titles):
        axes[0, i].set_title(title, fontsize=subtitle_fontsize, pad=20)  # Apply to the first row for column titles

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)  # Make space for titles and labels

    # Display the combined plot
    plt.show()



def plot_mean(all_runs_groups, names, dataset_labels):
    """
    Plots the mean cumulative maximum with 95% confidence intervals for multiple datasets, 
    grouped into two subplots, each corresponding to one of two names. Each dataset in the 
    group has its own mean and confidence interval plotted in the same subplot.

    Parameters:
    - all_runs_groups: List of two lists, where each sublist contains multiple datasets (each dataset is a list of runs).
                       Example: [[runs_group1_dataset1, runs_group1_dataset2], [runs_group2_dataset1, runs_group2_dataset2]]
    - names: List of two strings, each representing the name of the corresponding group in all_runs_groups.
             Example: ["Condition 1", "Condition 2"]
    - dataset_labels: List of two lists, where each sublist contains labels for the corresponding datasets in all_runs_groups.
                      Example: [["Dataset 1 Label", "Dataset 2 Label"], ["Dataset 1 Label", "Dataset 2 Label"]]
    """
    if len(all_runs_groups) != 2 or len(names) != 2 or len(dataset_labels) != 2:
        raise ValueError("all_runs_groups, names, and dataset_labels must each contain exactly two elements.")

    # Define font sizes
    title_fontsize = 20
    label_fontsize = 16
    tick_fontsize = 14
    legend_fontsize = 14

    # Create subplots with larger figure size for better readability
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

    for idx, (runs_group, name, labels) in enumerate(zip(all_runs_groups, names, dataset_labels)):
        ax = axes[idx]

        if not runs_group:
            ax.set_title(name, fontsize=title_fontsize)
            ax.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=label_fontsize)
            ax.axis('off')
            continue

        for dataset_idx, (runs, dataset_label) in enumerate(zip(runs_group, labels)):

            # Determine the maximum number of iterations across all runs for this dataset
            max_iterations = max(len(run) for run in runs)

            # Initialize a list to store cumulative maxima for each run
            cumulative_max_runs = []

            for run in runs:
                cumulative_max = []
                current_max = -np.inf
                for val in run:
                    if val > current_max:
                        current_max = val
                    cumulative_max.append(current_max)
                # If the run has fewer iterations than max_iterations, extend with the last max
                if len(cumulative_max) < max_iterations:
                    last_max = cumulative_max[-1]
                    cumulative_max.extend([last_max] * (max_iterations - len(cumulative_max)))
                cumulative_max_runs.append(cumulative_max)

            # Convert to a NumPy array for easier manipulation
            cumulative_max_array = np.array(cumulative_max_runs)  # Shape: (num_runs, max_iterations)

            # Calculate mean and confidence intervals
            mean_values = np.mean(cumulative_max_array, axis=0)
            sem_values = stats.sem(cumulative_max_array, axis=0)
            confidence = 0.95  # 95% confidence interval
            z_score = stats.norm.ppf(1 - (1 - confidence) / 2)  # Two-tailed

            margin_of_error = z_score * sem_values
            lower_bounds = mean_values - margin_of_error
            upper_bounds = mean_values + margin_of_error

            # Define iterations for plotting
            iterations = np.arange(1, max_iterations + 1)

            # Use provided dataset label instead of 'Dataset X'
            ax.plot(iterations, 100 * mean_values, label=f'{dataset_label}', linewidth=2)

            # Plot the confidence interval as a shaded area
            ax.fill_between(iterations, 100 * lower_bounds, 100 * upper_bounds,  alpha=0.3, label=f'95% CI {dataset_label}')

        # Optional: Plot a reference horizontal line (e.g., Goemans-Williamson Threshold)
        ax.axhline(y=87.8, color='r', linestyle='--', label='Goemans-Williamson Threshold')
        
        # Set titles and labels with increased font sizes
        ax.set_title(name, fontsize=title_fontsize)
        ax.set_xlabel("Iteration", fontsize=label_fontsize)
        if idx == 0:
            ax.set_ylabel("Approximation Ratio (%)", fontsize=label_fontsize)
        
        # Customize tick parameters for larger font sizes
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        # Set grid, limits, and legend
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim(50, 100)  
        ax.set_xlim(0, 500)  # Adjust x-limit based on data
        ax.legend(loc='upper left', fontsize=legend_fontsize, framealpha=0.9)
    
    # Optimize layout to prevent overlap
    plt.tight_layout()
    
    # Show the plot
    plt.show()


def plot_results2(results1, results2, folder_labels):
    """
    Plots the approximation ratios side by side for two datasets.

    Parameters:
    - results1 (dict): Processed data for the first folder.
    - results2 (dict): Processed data for the second folder.
    - folder_labels (tuple): Labels for the folders (for plot titles).
    """
    # Set up the subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

    def plot_runs(ax, runs, title):
        """
        Helper function to plot approximation ratios on a given axis.
        """
        for training_history in runs:
            iterations = list(range(1, len(training_history) + 1))
            ax.plot(iterations, 100 * np.array(training_history), alpha=0.3, color="#1f77b4")

        ax.axhline(y=87.8, color='r', linestyle='--', label='Goemans-Williamson Threshold')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Iteration", fontsize=14)
        ax.set_ylabel("Approximation Ratio (%)", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylim(50, 100)
        ax.set_xlim(0, 500)
        ax.grid(True)
        ax.legend(loc='upper left', fontsize=12)

    # Plot for the first dataset
    plot_runs(axes[0], results1['all_runs'], folder_labels[0])

    # Plot for the second dataset
    plot_runs(axes[1], results2['all_runs'], folder_labels[1])

    # Optimize layout
    plt.tight_layout()

    # Show the plots
    plt.show()


def process_folder2(folder_path, edit_dist, solution_similarities_min, solution_similarities_mean):
    """
    Processes a single folder using associated data lists.

    Returns a dictionary with processed data.
    """
    # Initialize counters
    total_files = 0
    better_than_0878 = 0
    instances_in_local_minima = 0
    instances_stuck_on_BP = 0
    instances_which_converged = 0

    # Initialize lists to store values for later analysis
    edit_dists = []
    hamming_dists_min = []
    hamming_dists_mean = []
    seeds = []

    # List to store approximation ratios for plotting
    all_runs = []

    # List all .txt files and sort them to ensure consistent ordering
    txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])

    # Iterate through each file and its corresponding data
    for idx, filename in enumerate(txt_files):
        filepath = os.path.join(folder_path, filename)

        # Initialize variables to store data
        maxcut = None
        qaoa_values = []
        seed = None

        with open(filepath, 'r') as file:
            for line in file:
                # Extract the graph seed
                if line.startswith("200 iters for seed") or line.startswith("500 iters for seed"):
                    try:
                        parts = line.split("seed")
                        seed_part = parts[1].split("on")[0].strip()
                        seed = int(seed_part)
                        seeds.append(seed)
                    except (IndexError, ValueError):
                        print(f"Warning: Could not parse seed in file {filename}")
                        seed = None

                # Extract the Maxcut value
                elif line.startswith(" Maxcut:"):
                    try:
                        maxcut = float(line.split()[1].strip(','))
                    except (IndexError, ValueError):
                        print(f"Warning: Could not parse Maxcut in file {filename}")
                        maxcut = None

                # Collect training data
                elif line.strip().replace('-', '').replace('.', '').isdigit():
                    try:
                        qaoa_value = -float(line.strip())  # Remove additional negative as a minimizer was used
                        qaoa_values.append(qaoa_value)
                    except ValueError:
                        print(f"Warning: Could not parse iteration data in file {filename}")

        # Compute Approximation Ratio at each iteration
        if maxcut and qaoa_values:
            training_history = [val / maxcut for val in qaoa_values]
            all_runs.append(training_history)
            iterations = list(range(1, len(training_history) + 1))

            # Check if the last normalized value is better than 0.878
            if training_history[-1] > 0.878:
                better_than_0878 += 1
            if training_history[-1] >0.9999:
                instances_which_converged += 1
            if len(training_history) < 20:
                instances_in_local_minima += 1
            elif training_history[-1] <0.75:
                instances_stuck_on_BP+= 1

            total_files += 1

            # Store the edit and Hamming distances for each graph as well as the final approximation ratio
            # Since data is provided as lists aligned with files, use the current index
            edit_dists.append([edit_dist[seed], training_history[-1]])
            hamming_dists_min.append([solution_similarities_min[seed], training_history[-1]])
            hamming_dists_mean.append([solution_similarities_mean[seed], training_history[-1]])
        else:
            print(f"Warning: No valid QAOA values or Maxcut found in file {filename}")

    # Calculate the fraction of instances with a better approximation ratio than 0.878
    fraction_better_than_0878 = better_than_0878 / total_files if total_files > 0 else 0
    print(f"Folder '{folder_path}':")
    print(f"  Total files processed: {total_files}")
    print(f"  Instances better than 0.878: {better_than_0878}")
    print(f"  Fraction better than 0.878: {fraction_better_than_0878:.4f}")
    print(f"  Instances in local minima: {instances_in_local_minima}")
    print(f"  Instances stuck on BP: {instances_stuck_on_BP}")
    print(f"  Instances which converged: {instances_which_converged}\n")

    # Prepare counters dictionary
    counters = {
        'total_files': total_files,
        'better_than_0878': better_than_0878,
        'fraction_better_than_0878': fraction_better_than_0878,
        'instances_in_local_minima': instances_in_local_minima,
        'instances_stuck_on_BP': instances_stuck_on_BP,
        'instances_which_converged': instances_which_converged
    }

    return {
        'edit_dists': edit_dists,
        'hamming_dists_min': hamming_dists_min,
        'hamming_dists_mean': hamming_dists_mean,
        'counters': counters,
        'all_runs': all_runs,
        'seeds': seeds
    }

def plot_and_analyze_results(
    folder_path1,
    edit_dist_1,
    solution_similarities_min_1,
    solution_similarities_mean_1,
    folder_path2,
    edit_dist_2,
    solution_similarities_min_2,
    solution_similarities_mean_2
):
    """
    Processes QAOA result files from two folders with their associated data lists, 
    plots approximation ratios side by side, and returns relevant data for both folders.

    Parameters:
    - folder_path1 (str): Path to the first folder containing the .txt files.
    - edit_dist_ER30_1 (list): List of edit distances for the first folder.
    - solution_similarities_min_ER30_1 (list): List of minimum Hamming distances for the first folder.
    - solution_similarities_mean_ER30_1 (list): List of mean Hamming distances for the first folder.
    - folder_path2 (str): Path to the second folder containing the .txt files.
    - edit_dist_ER30_2 (list): List of edit distances for the second folder.
    - solution_similarities_min_ER30_2 (list): List of minimum Hamming distances for the second folder.
    - solution_similarities_mean_ER30_2 (list): List of mean Hamming distances for the second folder.

    Returns:
    - results1 (dict): Contains edit_dists, hamming_dists_min, hamming_dists_mean, and counters for folder1.
    - results2 (dict): Contains edit_dists, hamming_dists_min, hamming_dists_mean, and counters for folder2.
    """

    def process_folder(folder_path, edit_dist, solution_similarities_min, solution_similarities_mean):
        """
        Helper function to process a single folder using associated data lists.

        Returns a dictionary with processed data.
        """
        # Initialize counters
        total_files = 0
        better_than_0878 = 0

        # Initialize lists to store values for later analysis
        edit_dists = []
        hamming_dists_min = []
        hamming_dists_mean = []

        # List to store approximation ratios for plotting
        all_runs = []

        # List all .txt files and sort them to ensure consistent ordering
        txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])


        # Iterate through each file and its corresponding data
        for idx, filename in enumerate(txt_files):
            filepath = os.path.join(folder_path, filename)

            # Initialize variables to store data
            maxcut = None
            qaoa_values = []
            seed= None

            with open(filepath, 'r') as file:
                for line in file:
                    # Extract the graph seed
                    if line.startswith("200 iters for seed") or line.startswith("500 iters for seed"):
                        try:
                            parts = line.split("seed")
                            seed_part = parts[1].split("on")[0].strip()
                            seed = int(seed_part)
                        except (IndexError, ValueError):
                            print(f"Warning: Could not parse seed in file {filename}")
                            seed = None

                    # Extract the Maxcut value
                    elif line.startswith(" Maxcut:"):
                        try:
                            maxcut = float(line.split()[1].strip(','))
                        except (IndexError, ValueError):
                            print(f"Warning: Could not parse Maxcut in file {filename}")
                            maxcut = None

                    # Collect training data
                    elif line.strip().replace('-', '').replace('.', '').isdigit():
                        try:
                            qaoa_value = -float(line.strip())  # Remove additional negative as a minimizer was used
                            qaoa_values.append(qaoa_value)
                        except ValueError:
                            print(f"Warning: Could not parse iteration data in file {filename}")

            # Compute Approximation Ratio at each iteration
            if maxcut and qaoa_values:
                training_history = [val / maxcut for val in qaoa_values]
                all_runs.append(training_history)
                iterations = list(range(1, len(training_history) + 1))

                # Check if the last normalized value is better than 0.878
                if training_history[-1] > 0.878:
                    better_than_0878 += 1

                total_files += 1

                # Store the edit and Hamming distances for each graph as well as the final approximation ratio
                # Since data is provided as lists aligned with files, use the current index
                edit_dists.append([edit_dist[seed], training_history[-1]])
                hamming_dists_min.append([solution_similarities_min[seed], training_history[-1]])
                hamming_dists_mean.append([solution_similarities_mean[seed], training_history[-1]])
            else:
                print(f"Warning: No valid QAOA values or Maxcut found in file {filename}")

        # Calculate the fraction of instances with a better approximation ratio than 0.878
        fraction_better_than_0878 = better_than_0878 / total_files if total_files > 0 else 0
        print(f"Folder '{folder_path}':")
        print(f"  Total files processed: {total_files}")
        print(f"  Instances better than 0.878: {better_than_0878}")
        print(f"  Fraction better than 0.878: {fraction_better_than_0878:.4f}\n")

        # Prepare counters dictionary
        counters = {
            'total_files': total_files,
            'better_than_0878': better_than_0878,
            'fraction_better_than_0878': fraction_better_than_0878
        }

        return {
            'edit_dists': edit_dists,
            'hamming_dists_min': hamming_dists_min,
            'hamming_dists_mean': hamming_dists_mean,
            'counters': counters,
            'all_runs': all_runs
        }

    # Process both folders with their respective data lists
    results1 = process_folder(folder_path1, edit_dist_1, solution_similarities_min_1, solution_similarities_mean_1)
    results2 = process_folder(folder_path2, edit_dist_2, solution_similarities_min_2, solution_similarities_mean_2)

    # Set up the subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

    def plot_runs(ax, runs,graph):     #, folder_label):
        """
        Helper function to plot approximation ratios on a given axis.
        """
        for training_history in runs:
            iterations = list(range(1, len(training_history) + 1))
            ax.plot(iterations, 100 * np.array(training_history), alpha=0.3, color="#1f77b4")

        ax.axhline(y=87.8, color='r', linestyle='--', label='Goemans-Williamson Threshold')
        if graph ==0:
            ax.set_title("16 Vertex Erdros-Renyi Graph (Edge probability 30%)", fontsize=16)
        elif graph ==1:
            ax.set_title("16 Vertex 3-Regular Graph", fontsize=16)
        ax.set_xlabel("Iteration", fontsize=14)
        ax.set_ylabel("Approximation Ratio (%)", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylim(50, 100)
        ax.set_xlim(0, 200)
        ax.grid(True)
        ax.legend(loc='upper left', fontsize=12)

    # Plot for first folder
    plot_runs(axes[0], results1['all_runs'],0)  #, os.path.basename(folder_path1))

    # Plot for second folder
    plot_runs(axes[1], results2['all_runs'],1)   #  , os.path.basename(folder_path2))

    # Optimize layout
    plt.tight_layout()

    # Show the plots
    plt.show()

    return results1, results2


def process_percentiles(folder_path):
    # Initialize lists to store percentiles and bad instances
    percentiles = []
    # List to store all normalized runs
    all_normalized_runs = []
    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            
            # Initialize variables to store data
            maxcut = None
            qaoa_values = []
            percentile = None  # Variable to store percentile
            
            with open(filepath, 'r') as file:
                for line in file:
                    # Extract the percentile from the first line (it's usually there)
                    if line.startswith("200 iters for seed") or line.startswith("500 iters for seed") and "percentile" in line:
                        try:
                            # Split the line using 'percentile' and extract the number
                            l1 = line.split("the")
                            percentile_str = l1[1].split("percentile")[0].strip()
                            if percentile_str:  # Ensure there's something to convert
                                percentile = float(percentile_str)
                            l2 = line.split("seed")
                            seed = l2[1].split("on")[0].strip()
                            s=int(seed)
                        except (IndexError, ValueError):
                            print(f"Warning: Could not parse percentile in file {filename}")
                            percentile = None

                    # Extract the Maxcut value
                    elif line.startswith(" Maxcut:"):
                        try:
                            maxcut = float(line.split()[1].strip(','))
                        except (IndexError, ValueError):
                            print(f"Warning: Could not parse Maxcut in file {filename}")
                            maxcut = None
                            
                    # Collect QAOA values if the line starts with a number (i.e., it's not metadata)
                    elif line.strip().replace('-', '').replace('.', '').isdigit():
                        try:
                            qaoa_values.append(-float(line.strip()))
                        except ValueError:
                            print(f"Warning: Could not parse QAOA value in file {filename}")

            # Normalize QAOA values by Maxcut and prepare iterations
            normalized_values = [val / maxcut for val in qaoa_values]
                
            percentiles.append([percentile,normalized_values[-1],s])
    return percentiles
