





import os


main_dir_local = "/Users/eliapizzati/projects/swift_qso/swift_smbh_evolution"
plot_dir = os.path.join(main_dir_local,"plots")
out_dir_local = os.path.join(main_dir_local, "outputs")

data_dir_local = "/Users/eliapizzati/projects/swift_qso/data"

# the following is meant to be for the cosma machine in durham
out_dir_machine_cosma = "/cosma8/data/dp004/dc-pizz1/projects/swift_qso/outputs"
plot_dir_machine_cosma = "/cosma8/data/dp004/dc-pizz1/projects/swift_qso/plots"
data_dir_machine_cosma = "/cosma8/data/dp004/dc-pizz1/projects/swift_qso/data"

# the following is meant to be for the igm machine in leiden
out_dir_machine_igm = "/data3/pizzati/projects/swift_qso/output_data/bh_evolution"
plot_dir_machine_igm = "/data3/pizzati/projects/swift_qso/plots/bh_evolution"
data_dir_machine_igm = "/data3/pizzati/projects/swift_qso/data/"




# functions to get paths
def get_input_path_HBT_data(source="local"):
    """
    Returns the path to the input data for HBT runs. 

    Parameters:
    source (str): The source of the data. Options are 'local', 'machine_cosma', or 'machine_igm'.

    Returns:
    str: The path to the input data directory.
    """

    if source == "local":
        data_dir = os.path.join(data_dir_local, "HBT_runs_FLAMINGO")
    elif source == "machine_cosma":
        data_dir = os.path.join(data_dir_machine_cosma, "HBT_runs_FLAMINGO")
    elif source == "machine_igm":
        data_dir = os.path.join(data_dir_machine_igm, "HBT_runs_FLAMINGO")
    else:
        raise ValueError("Invalid source. Choose from 'local', 'machine_cosma', or 'machine_igm'.")

    return data_dir


def get_output_path(source="local"):
    """
    Returns the output path based on the source.

    Parameters:
    source (str): The source of the output. Options are 'local', 'machine_cosma', or 'machine_igm'.

    Returns:
    str: The output directory path.
    """

    if source == "local":
        return out_dir_local
    elif source == "machine_cosma":
        return out_dir_machine_cosma
    elif source == "machine_igm":
        return out_dir_machine_igm
    else:
        raise ValueError("Invalid source. Choose from 'local', 'machine_cosma', or 'machine_igm'.")
    


def get_plots_path(source="local"):
    """
    Returns the plots path based on the source.

    Parameters:
    source (str): The source of the plots. Options are 'local', 'machine_cosma', or 'machine_igm'.

    Returns:
    str: The plots directory path.
    """

    if source == "local":
        return plot_dir
    elif source == "machine_cosma":
        return plot_dir_machine_cosma
    elif source == "machine_igm":
        return plot_dir_machine_igm
    else:
        raise ValueError("Invalid source. Choose from 'local', 'machine_cosma', or 'machine_igm'.")



