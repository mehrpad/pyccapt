import matplotlib.pyplot as plt
import plotly
from pathlib import Path

from pyccapt.calibration.leap_tools import leap_tools


def pre_data(pos_file, rrng_file, input_type):
    """
    Preprocesses the data by reading the position (POS or EPOS) and range (RRNG) files,
    labeling ions, and performing deconvolution.

    Args:
        pos_file (str): Path to the position file (POS or EPOS).
        rrng_file (str): Path to the range file (RRNG).
        input_type (str): Type of input data ('pos' or 'epos').

    Returns:
        pandas.DataFrame: Preprocessed elements.

    """
    if input_type == 'pos':
        pos = leap_tools.read_pos(pos_file)
        ions, rrngs = leap_tools.read_rrng(rrng_file)
        pos_comp = leap_tools.label_ions(pos, rrngs)
        elements = leap_tools.deconvolve(pos_comp)

    elif input_type == 'epos':
        epos = leap_tools.read_epos(pos_file)
        ions, rrngs = leap_tools.read_rrng(rrng_file)
        epos_comp = leap_tools.label_ions(epos, rrngs)
        elements = leap_tools.deconvolve(epos_comp)

    return elements


def decompose(data, element):
    """
    Decomposes the data into separate coordinates and color for a specific element.

    Args:
        data (pandas.DataFrame): Data containing coordinates and colors.
        element (str): Element label.

    Returns:
        tuple: Decomposed coordinates and color.

    """
    initial = data.loc[element]
    pos = initial[['x (nm)', 'y (nm)', 'z (nm)']].values
    color = initial['colour'].values[0]
    px, py, pz = (pos[:, 0], pos[:, 1], pos[:, 2])
    return px, py, pz, color


def cloud_plotter(data, phases, result_path, filename, plot_type='cloud', open_new_window=False):
    """
    Generates a cloud plot or projection plot based on the data and desired plot type.

    Args:
        data (pandas.DataFrame): Data containing coordinates, colors, and elements.
        phases (list): List of elements to include in the plot.
        result_path (str): Path to save the output file.
        filename (str): Name of the output file.
        plot_type (str): Type of plot ('cloud' or 'projection').
        open_new_window (bool): Flag to open the plot in a new window.

    Returns:
        dict or None: Plotly figure dictionary if plot_type is 'cloud', otherwise None.

    """
    if plot_type == 'projection':
        # Plot static images with matplotlib.
        output_dir = Path(result_path).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        ax = plt.figure().add_subplot(111)
        for element in phases:
            px, py, pz, color = decompose(data, element)
            ax.scatter(py, pz, s=2, label=element)
        ax.xaxis.tick_top()
        ax.invert_yaxis()
        ax.set_xlabel('Y')
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('Z')
        plt.legend()
        plt.savefig(output_dir / f"output_{filename}.png")

    elif plot_type == 'cloud':
        # Adjust fig parameters in def_function.py
        plotly_data = list()
        for element in phases:
            px, py, pz, color = decompose(data, element)
            scatter = dict(
                mode="markers",
                name=element,
                type="scatter3d",
                x=px, y=py, z=pz,
                opacity=0.2,
                marker=dict(size=2, color=color)
            )
            plotly_data.append(scatter)

        layout = dict(
            title='APT 3D Point Cloud',
            scene=dict(xaxis=dict(zeroline=False, title='x (nm)'),
                       yaxis=dict(zeroline=False, title='y (nm)'),
                       zaxis=dict(zeroline=False, title='z (nm)', autorange='reversed'))
        )

        fig = dict(data=plotly_data, layout=layout)
        if open_new_window:
            output_dir = Path(result_path).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            plotly.offline.plot(fig, filename=str(output_dir / f"{filename}.html"), show_link=False)
        else:
            return fig

    else:
        print("Plot type error!")
