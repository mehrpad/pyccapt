import io
from copy import copy

import imageio
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import rcParams, colors
from matplotlib.animation import FuncAnimation
from PIL import Image
from plotly.subplots import make_subplots

# Local module and scripts
from pyccapt.calibration.data_tools import data_loadcrop, selectors_data


def cart2pol(x, y):
    """
    Convert Cartesian coordinates to polar coordinates.

    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.

    Returns:
        float: rho, the radial distance from the origin.
        float: phi, the angle in radians.
    """
    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    """
    Convert polar coordinates to Cartesian coordinates.

    Args:
        rho (float): radial distance from the origin.
        phi (float): angle in radians.

    Returns:
        float: x-coordinate.
        float: y-coordinate.
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def atom_probe_recons_from_detector_Gault_et_al(detx, dety, hv, flight_path_length, kf, det_eff, icf, field_evap, avg_dens):
    """
    Perform atom probe reconstruction using Geiser et al.'s method.

    Args:
        detx (float): Hit position on the detector (x-coordinate).
        dety (float): Hit position on the detector (y-coordinate).
        hv (float): High voltage.
        flight_path_length (float): Distance between detector and sample.
        kf (float): Field reduction factor.
        det_eff (float): Efficiency of the detector.
        icf (float): Image compression factor due to sample imperfections.
        field_evap (float): Evaporation field in V/nm.
        avg_dens (float): Atomic density in atoms/nm^3.

    Returns:
        float: x-coordinates of reconstructed atom positions in nm.
        float: y-coordinates of reconstructed atom positions in nm.
        float: z-coordinates of reconstructed atom positions in nm.
    """
    # Convert detector coordinates to polar form
    rad, ang = cart2pol(detx * 1E1, dety * 1E1)

    # Calculate effective detector area
    det_area = (np.max(rad) ** 2) * np.pi

    # Calculate radius evolution
    radius_evolution = hv / (kf * field_evap)

    # Calculate launch angle relative to specimen axis
    theta_p = np.arctan(rad / flight_path_length)

    # Calculate theta normal (image compression correction)
    theta_a = theta_p + np.arcsin((icf - 1) * np.sin(theta_p))

    # Convert polar coordinates to Cartesian coordinates
    z_p, d = pol2cart(radius_evolution, theta_a)
    x, y = pol2cart(d, ang)

    # Calculate z coordinate
    # the z shift with respect to the top of the cap is Rspec - zP
    # z_p = radius_evolution * (1 - np.cos(theta_a))
    z_p = radius_evolution - z_p
    omega = 1 / avg_dens
    omega = omega / det_eff

    M = flight_path_length / (icf * radius_evolution)

    spec_area = det_area / M ** 2

    dz = omega / spec_area
    # icf_2 = theta_a / theta_p
    # dz = (omega * ((flight_path_length * 1E-3) ** 2) * (kf ** 2) * ((field_evap / 1E-9) ** 2)) / (
    #         det_area * det_eff * (icf_2 ** 2) * (hv ** 2))

    # dz_o = (omega * ((flight_path_length * 1E-3) ** 2) * (kf ** 2) * ((field_evap / 1E-9) ** 2)) / (
    #         det_area * det_eff * (icf ** 2) * (hv ** 2))

    # dz = avg_dens * flight_path_length ** 2 * kf ** 2 * field_evap ** 2 / (det_area * det_eff * icf ** 2 * hv ** 2)

    # assert dz.any() != dz_o.any(), 'dz is not equal to dz_o'
    # check if dz has any NaN of inf values
    assert not np.isnan(dz).any(), 'dz has NaN values'
    assert not np.isinf(dz).any(), 'dz has inf values'

    cum_z = np.cumsum(dz)
    z = cum_z + z_p

    return x, y, z


def atom_probe_recons_Bas_et_al(detx, dety, hv, flight_path_length, kf, det_eff, icf, field_evap, avg_dens):
    """
    Perform atom probe reconstruction using Bas et al.'s method.

    Args:
        detx (float): Hit position on the detector (x-coordinate).
        dety (float): Hit position on the detector (y-coordinate).
        hv (float): High voltage.
        flight_path_length (float): Distance between detector and sample.
        kf (float): Field reduction factor.
        det_eff (float): Efficiency of the detector.
        icf (float): Image compression factor due to sample imperfections.
        field_evap (float): Evaporation field in V/nm.
        avg_dens (float): Atomic density in atoms/nm^3.

    Returns:
        float: x-coordinates of reconstructed atom positions in nm.
        float: y-coordinates of reconstructed atom positions in nm.
        float: z-coordinates of reconstructed atom positions in nm.
    """
    radius_evolution = hv / (kf * (field_evap / 1E-9))
    m = (flight_path_length * 1E-3) / (icf * radius_evolution)

    x = (detx * 1E-2) / m
    y = (dety * 1E-2) / m

    rad, ang = cart2pol(detx * 1E-3, dety * 1E-3)
    det_area = (np.max(rad) ** 2) * np.pi

    omega = 1E-9 ** 3 / avg_dens
    dz = (omega * ((flight_path_length * 1E-3) ** 2) * (kf ** 2) * ((field_evap / 1E-9) ** 2)) / (
            det_area * det_eff * (icf ** 2) * (hv ** 2))
    dz_p = radius_evolution * (1 - np.sqrt(1 - ((x ** 2 + y ** 2) / (radius_evolution ** 2))))
    z = np.cumsum(dz) + dz_p

    return x * 1E9, y * 1E9, z * 1E9


def draw_qube(fig, range, col=None, row=None):
    x_range = range[0]
    y_range = range[1]
    z_range = range[2]

    x_corner = [x_range[0], x_range[0], x_range[0], x_range[0], x_range[1], x_range[1], x_range[1], x_range[1]]
    y_corner = [y_range[0], y_range[0], y_range[1], y_range[1], y_range[0], y_range[0], y_range[1], y_range[1]]
    z_corner = [z_range[0], z_range[1], z_range[0], z_range[1], z_range[0], z_range[1], z_range[0], z_range[1]]

    edges = [(0, 1), (1, 5), (5, 4), (4, 0),  # Bottom edges
             (2, 3), (3, 7), (7, 6), (6, 2),  # Top edges
             (0, 2), (1, 3), (5, 7), (4, 6)]  # Vertical edges
    if col is not None or row is not None:
        for edge in edges:
            fig.add_trace(
                go.Scatter3d(
                    x=[x_corner[edge[0]], x_corner[edge[1]]],
                    y=[y_corner[edge[0]], y_corner[edge[1]]],
                    z=[z_corner[edge[0]], z_corner[edge[1]]],
                    mode='lines',
                    line=dict(color='black', width=2),
                    showlegend=False,
                    hoverinfo='none'
                ), row=row + 1, col=col + 1
            )
    else:
        for edge in edges:
            fig.add_trace(
                go.Scatter3d(
                    x=[x_corner[edge[0]], x_corner[edge[1]]],
                    y=[y_corner[edge[0]], y_corner[edge[1]]],
                    z=[z_corner[edge[0]], z_corner[edge[1]]],
                    mode='lines',
                    line=dict(color='black', width=2),
                    showlegend=False,
                    hoverinfo='none'
                )
            )
    # choose the figure font
    font_dict = dict(family='Arial',
                     size=10,
                     color='black'
                     )
    fig.update_layout(font=font_dict)
    fig.update_layout(
        scene=dict(
            xaxis_title="x [nm]",
            yaxis_title="y [nm]",
            zaxis_title="z [nm]",
        )
    )

    fig.update_scenes(zaxis_autorange="reversed")
    fig.update_layout(
        legend_title="",
        legend={'itemsizing': 'constant'},
        font=dict(size=8)
    )
    return fig


def reconstruction_plot(variables, element_percentage, opacity, rotary_fig_save, figname, save, make_gif=False,
                        make_evaporation_gif=False, range_sequence=[], range_mc=[], range_detx=[], range_dety=[],
                        range_x=[], range_y=[], range_z=[], range_vol=[], ions_individually_plots=False,
                        detailed_isotope_charge=False, colab=False):
    """
    Generate a 3D plot for atom probe reconstruction data.

    Args:
        variables (object): Variables object.
        element_percentage (str): Percentage of elements to display.
        opacity (float): Opacity of the markers.
        rotary_fig_save (bool): Whether to save the rotary figure.
        figname (str): Name of the figure.
        save (bool): Whether to save the figure.
        make_gif (bool): Whether to make a GIF.
        make_evaporation_gif (bool): Whether to make an evaporation GIF.
        range_sequence (list): Range of sequence
        range_mc: Range of mc
        range_detx: Range of detx
        range_dety: Range of dety
        range_x: Range of x-axis
        range_y: Range of y-axis
        range_z: Range of z-axis
        range_vol: Range of volume
        ions_individually_plots (bool): Whether to plot ions individually.
        detailed_isotope_charge (bool): Whether to plot detailed isotope and charge information.
        colab (bool): Whether to run in Google Colab.
    Returns:
        None
    """
    if range_sequence or range_detx or range_dety or range_mc or range_x or range_y or range_z:
        if range_sequence:
            mask_sequence = np.zeros_like(variables.dld_x_det, dtype=bool)
            mask_sequence[range_sequence[0]:range_sequence[1]] = True
        else:
            mask_sequence = np.ones_like(variables.dld_x_det, dtype=bool)
        if range_detx and range_dety:
            mask_det_x = (variables.dld_x_det < range_detx[1]) & (variables.dld_x_det > range_detx[0])
            mask_det_y = (variables.dld_y_det < range_dety[1]) & (variables.dld_y_det > range_dety[0])
            mask_det = mask_det_x & mask_det_y
        else:
            mask_det = np.ones(len(variables.dld_x_det), dtype=bool)
        if range_mc:
            mask_mc = (variables.mc_uc < range_mc[1]) & (variables.mc_uc > range_mc[0])
        else:
            mask_mc = np.ones(len(variables.mc), dtype=bool)
        if range_x and range_y and range_z:
            mask_x = (variables.x < range_x[1]) & (variables.x > range_x[0])
            mask_y = (variables.y < range_y[1]) & (variables.y > range_y[0])
            mask_z = (variables.z < range_z[1]) & (variables.z > range_z[0])
            mask_3d = mask_x & mask_y & mask_z
        else:
            mask_3d = np.ones(len(variables.x), dtype=bool)
        if range_vol:
            mask_vol = (variables.volume < range_vol[1]) & (variables.volume > range_vol[0])
        else:
            mask_vol = np.ones(len(variables.volume), dtype=bool)

        mask_f = mask_sequence & mask_det & mask_mc & mask_3d & mask_vol
        print('The number of data sequence:', len(mask_sequence[mask_sequence == True]))
        print('The number of data mc:', len(mask_mc[mask_mc == True]))
        print('The number of data det:', len(mask_det[mask_det == True]))
        print('The number of data 3d:', len(mask_3d[mask_3d == True]))
        print('The number of data after cropping:', len(mask_f[mask_f == True]))

    else:
        mask_f = np.ones(len(variables.x), dtype=bool)

    if isinstance(element_percentage, list):
        pass
    else:
        print('element_percentage should be a list')

    colors = variables.range_data['color'].tolist()
    mc_low = variables.range_data['mc_low'].tolist()
    mc_up = variables.range_data['mc_up'].tolist()
    ion = variables.range_data['ion'].tolist()

    # Draw an edge of cube around the 3D plot
    x_range = [min(variables.x), max(variables.x)]
    y_range = [min(variables.y), max(variables.y)]
    z_range = [min(variables.z), max(variables.z)]
    range_cube = [x_range, y_range, z_range]

    # Create a subplots with shared axes
    if ions_individually_plots:
        num_plots = len(ion)
        rows = (num_plots // 3) + (1 if num_plots % 3 != 0 else 0)
        cols = 3
        subplot_titles = ion
        # Generate the specs dictionary based on the number of rows and columns
        specs = [[{"type": "scatter3d", "rowspan": 1, "colspan": 1} for _ in range(cols)] for _ in range(rows)]

        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles,
                            start_cell="top-left", specs=specs)
        for row in range(rows):
            for col in range(cols):
                index = col + row * 3
                if index == len(ion):
                    break
                mask = (variables.mc > mc_low[index]) & (variables.mc < mc_up[index])
                mask = mask & mask_f
                size = int(len(mask[mask == True]) * float(element_percentage[index]))
                # Find indices where the original mask is True
                true_indices = np.where(mask)[0]
                # Randomly choose 100 indices from the true indices
                random_true_indices = np.random.choice(true_indices, size=size, replace=False)
                # Create a new mask with the same length as the original, initialized with False
                new_mask = np.full(len(variables.dld_t), False)
                # Set the selected indices to True in the new mask
                new_mask[random_true_indices] = True
                # Apply the new mask to the original mask
                mask = mask & new_mask

                scatter = go.Scatter3d(
                    x=variables.x[mask],
                    y=variables.y[mask],
                    z=variables.z[mask],
                    mode='markers',
                    name=ion[index],
                    showlegend=True,
                    marker=dict(
                        size=1,
                        color=colors[index],
                        opacity=opacity,
                    )
                )
                fig = draw_qube(fig, range_cube, col, row)

                fig.add_trace(scatter, row=row + 1, col=col + 1)
    else:
        fig = go.Figure()
        for index, elemen in enumerate(ion):
            mask = (variables.mc > mc_low[index]) & (variables.mc < mc_up[index])
            mask = mask & mask_f
            size = int(len(mask[mask == True]) * float(element_percentage[index]))
            # Find indices where the original mask is True
            true_indices = np.where(mask)[0]
            # Randomly choose 100 indices from the true indices
            random_true_indices = np.random.choice(true_indices, size=size, replace=False)
            # Create a new mask with the same length as the original, initialized with False
            new_mask = np.full(len(variables.dld_t), False)
            # Set the selected indices to True in the new mask
            new_mask[random_true_indices] = True
            # Apply the new mask to the original mask
            mask = mask & new_mask

            fig.add_trace(
                go.Scatter3d(
                    x=variables.x[mask],
                    y=variables.y[mask],
                    z=variables.z[mask],
                    mode='markers',
                    name=ion[index],
                    showlegend=True,
                    marker=dict(
                        size=1,
                        color=colors[index],
                        opacity=opacity,
                    )
                )
            )

        fig = draw_qube(fig, range_cube)

    if rotary_fig_save or make_gif:
        if not ions_individually_plots:
            rotary_fig(go.Figure(fig), variables, rotary_fig_save, make_gif, figname)
        else:
            print('Rotary figure is not available for ions_individually_plots=True')

    if make_evaporation_gif:
        num_events = len(variables.dld_t)
        figures = []
        for k in range(0, num_events, 100_000):
            rotated_fig = go.Figure()
            rotated_fig = draw_qube(rotated_fig, range_cube)
            rotated_fig.update_layout(showlegend=False)

            if k + 100_000 > num_events:
                q = num_events - 1
            else:
                q = k + 100_000
            mask_evap = (variables.dld_t > variables.dld_t[k]) & (variables.dld_t < variables.dld_t[q])

            for index, elemen in enumerate(ion):
                mask = (variables.mc > mc_low[index]) & (variables.mc < mc_up[index])
                mask = mask & mask_f & mask_evap
                size = int(len(mask[mask == True]) * float(element_percentage[index]))
                # Find indices where the original mask is True
                true_indices = np.where(mask)[0]
                # Randomly choose 100 indices from the true indices
                random_true_indices = np.random.choice(true_indices, size=size, replace=False)
                # Create a new mask with the same length as the original, initialized with False
                new_mask = np.full(len(variables.dld_t), False)
                # Set the selected indices to True in the new mask
                new_mask[random_true_indices] = True
                # Apply the new mask to the original mask
                mask = mask & new_mask

                rotated_fig.add_trace(
                    go.Scatter3d(
                        x=variables.x[mask],
                        y=variables.y[mask],
                        z=variables.z[mask],
                        mode='markers',
                        name=ion[index],
                        showlegend=True,
                        marker=dict(
                            size=1,
                            color=colors[index],
                            opacity=opacity,
                        )
                    )
                )
            print(' Plotted the ions up to the event:', q)
            figures.append(rotated_fig)

        images = []
        print('Starting to process the frames for the GIF')
        print('The total number of frames is:', len(figures))
        for index, frame in enumerate(figures):
            images.append(plotly_fig2array(frame))
            print('frame', index, 'is being processed')
        print('The images are ready for the GIF')

        # Save the images as a GIF using imageio
        imageio.mimsave(variables.result_path + '\\rota_evaporation_{fn}.gif'.format(fn=figname), images, fps=2)

    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.99
        )
    )

    config = dict(
        {
            'scrollZoom': True,
            'displayModeBar': True,
            'modeBarButtonsToAdd': [
                'drawline',
                'drawopenpath',
                'drawclosedpath',
                'drawcircle',
                'drawrect',
                'eraseshape'
            ]
        }
    )

    # x_range = [min(variables.x), max(variables.x)]
    # y_range = [min(variables.y), max(variables.y)]
    # z_range = [min(variables.z), max(variables.z)]
    #
    # x_len = x_range[1] - x_range[0]
    # y_len = y_range[1] - y_range[0]
    # z_len = z_range[1] - z_range[0]
    #
    # aspect_ratio = dict(x=x_len, y=y_len, z=z_len)
    #
    # fig.update_layout(scene=dict(aspectratio=aspect_ratio))

    fig.update_layout(scene=dict(aspectmode='data', aspectratio=dict(x=1, y=1, z=1)))
    fig.update_layout(
        scene=dict(  # Target the 3D scene
            bgcolor='white',  # Set scene background to white
            xaxis=dict(
                backgroundcolor="rgb(255,255,255)",
                gridcolor="lightgrey",
            ),
            yaxis=dict(
                backgroundcolor="rgb(255,255,255)",
                gridcolor="lightgrey"
            ),
            zaxis=dict(
                backgroundcolor="rgb(255,255,255)",
                gridcolor="lightgrey"
            )))

    # Show the plot in the Jupyter cell output
    variables.plotly_3d_reconstruction = go.FigureWidget(fig)


    if not colab:
        pio.renderers.default = 'browser'
        fig.show(config=config)
    else:
        fig.show(config=config)

    if save:
        try:
            # fig1 = go.Figure(fig)
            fig.update_scenes(
                camera=dict(
                    eye=dict(x=4, y=4, z=4),  # Adjust the camera position for zooming
                )
            )
            pio.write_html(fig, variables.result_path + "/%s_3d.html" % figname, include_mathjax='cdn')
            fig.update_layout(showlegend=False)
            layout = go.Layout(
                margin=go.layout.Margin(
                    l=0,  # left margin
                    r=0,  # right margin
                    b=0,  # bottom margin
                    t=0,  # top margin
                )
            )
            fig.update_layout(layout)
            pio.write_image(fig, variables.result_path + "/%s_3d.png" % figname, scale=3, format='png')
            pio.write_image(fig, variables.result_path + "/%s_3d.svg" % figname, scale=3, format='svg')
            fig.update_layout(showlegend=True)

            fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
            fig.update_layout(showlegend=False)
            pio.write_image(fig, variables.result_path + "/%s_3d_o.png" % figname, scale=3, format='png')
            pio.write_image(fig, variables.result_path + "/%s_3d_o.svg" % figname, scale=3, format='svg')
            fig.update_layout(showlegend=True)
            pio.write_html(fig, variables.result_path + "/%s_3d_o.html" % figname, include_mathjax='cdn')
            fig.update_scenes(xaxis_visible=True, yaxis_visible=True, zaxis_visible=True)
        except Exception as e:
            print('The figure could not be saved')
            print(e)



def rotate_z(x, y, z, theta):
    """
    Rotate coordinates around the z-axis.

    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.
        theta (float): Rotation angle.

    Returns:
        tuple: Rotated coordinates (x, y, z).
    """
    w = x + 1j * y
    return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z


def plotly_fig2array(fig):
    """
    convert Plotly fig to  an array

    Args:
        fig (plotly.graph_objects.Figure): The base figure.

    Returns:
        array: The array representation of the figure.
    """
    # convert Plotly fig to  an array
    fig_bytes = pio.to_image(fig, format="jpeg", scale=5, engine="kaleido")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


def rotary_fig(fig, variables, rotary_fig_save, make_gif, figname):
    """
    Generate a rotating figure using Plotly.

    Args:
        fig (plotly.graph_objects.Figure): The base figure.
        variables (object): The variables object.
        rotary_fig_save (bool): Whether to save the rotary figure.
        make_gif (bool): Whether to make a GIF.
        figname (str): The name of the figure.

    Returns:
        None
    """
    x_eye = -1.25
    y_eye = 2
    z_eye = 0.5
    fig = go.Figure(fig)

    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)

    if make_gif:
        fig.update_layout(showlegend=False)
        layout = go.Layout(
            margin=go.layout.Margin(
                l=0,  # left margin
                r=0,  # right margin
                b=0,  # bottom margin
                t=0,  # top margin
            )
        )
        fig.update_layout(layout)

        figures = []
        for t in np.arange(0, 4, 0.2):
            xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, t)
            rotated_fig = go.Figure(fig)
            rotated_fig.update_layout(scene_camera_eye=dict(x=xe, y=ye, z=ze))
            figures.append(rotated_fig)

        images = []
        print('Starting to process the frames for the GIF')
        print('The total number of frames is:', len(figures))
        for index, frame in enumerate(figures):
            images.append(plotly_fig2array(frame))
            print('frame', index, 'is being processed')
        print('The images are ready for the GIF')

        # Save the images as a GIF using imageio
        imageio.mimsave(variables.result_path + '\\rota_{fn}.gif'.format(fn=figname), images, fps=2)

        fig.update_layout(showlegend=True)

    if rotary_fig_save:
        fig.update_layout(
            scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    y=1.2,
                    x=0.8,
                    xanchor='left',
                    yanchor='bottom',
                    pad=dict(t=45, r=10),
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=15, redraw=True),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                    mode='immediate'
                                )
                            ]
                        )
                    ]
                )
            ]
        )

        frames = []

        for t in np.arange(0, 50, 0.1):
            xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
            frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))
        fig.frames = frames

        plotly.offline.plot(
            fig,
            filename=variables.result_path + '\\rota_{fn}.html'.format(fn=figname),
            show_link=True,
            auto_open=False,
            include_mathjax='cdn'
        )


def scatter_plot(data, range_data, variables, element_percentage, selected_area, x_or_y, figname, figure_size,
                 save=False):
    """
    Generate a scatter plot based on the provided data.

    Args:
        data (pandas.DataFrame): The input data.
        range_data (pandas.DataFrame): Data containing range information for different elements.
        variables (object): The variables object.
        element_percentage (str): Element percentage information.
        selected_area (bool): True if a specific area is selected, False otherwise.
        x_or_y (str): Either 'x' or 'y' indicating the axis to plot.
        figname (str): The name of the figure.

    Returns:
        None
    """
    fig = plt.figure(figsize=figure_size)  # Specify the width and height
    ax = fig.add_subplot(111)

    phases = range_data['element'].tolist()
    colors = range_data['color'].tolist()
    mc_low = range_data['mc_low'].tolist()
    mc_up = range_data['mc_up'].tolist()
    charge = range_data['charge'].tolist()
    isotope = range_data['isotope'].tolist()

    element_percentage = element_percentage.replace('[', '')
    element_percentage = element_percentage.replace(']', '')
    element_percentage = element_percentage.split(',')

    for index, elemen in enumerate(phases):
        df_s = data.copy(deep=True)
        df_s = df_s[(df_s['mc_c (Da)'] > mc_low[index]) & (df_s['mc_c (Da)'] < mc_up[index])]
        df_s.reset_index(inplace=True, drop=True)
        remove_n = int(len(df_s) - (len(df_s) * float(element_percentage[index])))
        drop_indices = np.random.choice(df_s.index, remove_n, replace=False)
        df_subset = df_s.drop(drop_indices)
        if phases[index] == 'unranged':
            name_element = 'unranged'
        else:
            name_element = r'${}^{%s}%s^{%s+}$' % (isotope[index], phases[index], charge[index])
        if x_or_y == 'x':
            ax.scatter(df_subset['x (nm)'], df_subset['z (nm)'], s=0.1, label=name_element, color=colors[index])
        elif x_or_y == 'y':
            ax.scatter(df_subset['y (nm)'], df_subset['z (nm)'], s=0.1, label=name_element)

    if not selected_area:
        data_loadcrop.rectangle_box_selector(ax, variables)
        plt.connect('key_press_event', selectors_data.toggle_selector)
    ax.xaxis.tick_top()
    ax.invert_yaxis()
    if x_or_y == 'x':
        ax.set_xlabel('x (nm)')
    elif x_or_y == 'y':
        ax.set_xlabel('y (nm)')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('z (nm)')
    plt.legend(loc='upper right')

    if save:
        # Enable rendering for text elements
        rcParams['svg.fonttype'] = 'none'
        plt.savefig(variables.result_path + '\\projection_{fn}.png'.format(fn=figname))
        plt.savefig(variables.result_path + '\\projection_{fn}.svg'.format(fn=figname))
    plt.show()


def projection(variables, element_percentage, range_sequence=[], range_mc=[], range_detx=[], range_dety=[], range_x=[],
               range_y=[], range_z=[], range_vol=[], x_or_y='x', figname='projection', figure_size=(5, 5), save=False):
    """
    Generate a projection plot based on the provided data.

    Args:
        variables (object): The variables object.
        element_percentage (str): Element percentage information.
        range_sequence: Range of sequence
        range_mc: Range of mc
        range_detx: Range of detx
        range_dety: Range of dety
        range_x: Range of x-axis
        range_y: Range of y-axis
        range_z: Range of z-axis
        range_vol: Range of volume
        x_or_y (str): Either 'x' or 'y' indicating the axis to plot.
        figname (str): The name of the figure.
    Returns:
        None
    """
    fig = plt.figure(figsize=figure_size)  # Specify the width and height
    ax = fig.add_subplot(111)

    if range_sequence or range_mc or range_detx or range_dety or range_x or range_y or range_z:
        if range_sequence:
            if range_sequence:
                mask_sequence = np.zeros_like(variables.dld_x_det, dtype=bool)
                mask_sequence[range_sequence[0]:range_sequence[1]] = True
            else:
                mask_sequence = np.ones_like(variables.dld_x_det, dtype=bool)
        if range_detx and range_dety:
            mask_det_x = (variables.dld_x_det < range_detx[1]) & (variables.dld_x_det > range_detx[0])
            mask_det_y = (variables.dld_y_det < range_dety[1]) & (variables.dld_y_det > range_dety[0])
            mask_det = mask_det_x & mask_det_y
        else:
            mask_det = np.ones(len(variables.dld_x_det), dtype=bool)
        if range_mc:
            mask_mc = (variables.mc < range_mc[1]) & (variables.mc > range_mc[0])
        else:
            mask_mc = np.ones(len(variables.mc), dtype=bool)
        if range_x and range_y and range_z:
            mask_x = (variables.x < range_x[1]) & (variables.x > range_x[0])
            mask_y = (variables.y < range_y[1]) & (variables.y > range_y[0])
            mask_z = (variables.z < range_z[1]) & (variables.z > range_z[0])
            mask_3d = mask_x & mask_y & mask_z
        else:
            mask_3d = np.ones(len(variables.x), dtype=bool)
        if range_vol:
            mask_vol = (variables.volume < range_vol[1]) & (variables.volume > range_vol[0])
        else:
            mask_vol = np.ones(len(variables.volume), dtype=bool)
        mask = mask_sequence & mask_det & mask_mc & mask_3d & mask_vol
        print('The number of data sequence:', len(mask_sequence[mask_sequence == True]))
        print('The number of data mc:', len(mask_mc[mask_mc == True]))
        print('The number of data det:', len(mask_det[mask_det == True]))
        print('The number of data 3d:', len(mask_3d[mask_3d == True]))
        print('The number of data after cropping:', len(mask[mask == True]))
    else:
        mask = np.ones(len(variables.mc_uc), dtype=bool)


    ions = variables.range_data['ion'].tolist()
    colors = variables.range_data['color'].tolist()
    mc_low = variables.range_data['mc_low'].tolist()
    mc_up = variables.range_data['mc_up'].tolist()


    if isinstance(element_percentage, list):
        pass
    else:
        print('element_percentage should be a list')


    for index, elemen in enumerate(ions):
        mask_spacial = (variables.mc > mc_low[index]) & (variables.mc < mc_up[index])
        mask = mask & mask_spacial
        size = int(len(mask[mask == True]) * float(element_percentage[index]))
        # Find indices where the original mask is True
        true_indices = np.where(mask)[0]
        # Randomly choose 100 indices from the true indices
        random_true_indices = np.random.choice(true_indices, size=size, replace=False)
        # Create a new mask with the same length as the original, initialized with False
        new_mask = np.full(len(variables.dld_t), False)
        # Set the selected indices to True in the new mask
        new_mask[random_true_indices] = True
        # Apply the new mask to the original mask
        mask = mask & new_mask
        if ions[index] == 'unranged':
            name_element = 'unranged'
        else:
            name_element = '%s' % ions[index]
        if x_or_y == 'x':
            ax.scatter(variables.x[mask], variables.z[mask], s=0.1,
                       label=name_element, color=colors[index])
        elif x_or_y == 'y':
            ax.scatter(variables.y[mask], variables.z[mask], s=0.1,
                       label=name_element, color=colors[index])

    # ax.xaxis.tick_top()
    ax.invert_yaxis()
    if x_or_y == 'x':
        ax.set_xlabel('x (nm)')
    elif x_or_y == 'y':
        ax.set_xlabel('y (nm)')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('z (nm)')
    plt.legend(loc='upper right')

    if save:
        # Enable rendering for text elements
        rcParams['svg.fonttype'] = 'none'
        plt.savefig(variables.result_path + '\\projection_{fn}.png'.format(fn=figname), format="png", dpi=600)
        plt.savefig(variables.result_path + '\\projection_{fn}.svg'.format(fn=figname), format="svg", dpi=600)
    plt.show()


def heatmap(variables, element_percentage, range_sequence=[], range_mc=[], range_detx=[], range_dety=[], range_x=[],
            range_y=[], range_z=[], range_vol=[], figure_name='hetmap', figure_sie=(5, 5), save=False):
    """
    Generate a heatmap based on the provided data.

    Args:
        variables (object): The variables object.
        element_percentage (str): Element percentage information.
        range_sequence: Range of sequence
        range_mc: Range of mc
        range_detx: Range of detx
        range_dety: Range of dety
        range_x: Range of x-axis
        range_y: Range of y-axis
        range_z: Range of z-axis
        range_vol: Range of volume
        figure_name (str): The name of the figure.
        figure_sie: The size of the figure.
        save (bool): True to save the plot, False to display it.

    Returns:
        None
    """
    fig = plt.figure(figsize=figure_sie)  # Specify the width and height
    ax = fig.add_subplot(111)

    if range_sequence or range_mc or range_detx or range_dety or range_x or range_y or range_z:
        if range_sequence:
            if range_sequence:
                mask_sequence = np.zeros_like(variables.dld_x_det, dtype=bool)
                mask_sequence[range_sequence[0]:range_sequence[1]] = True
            else:
                mask_sequence = np.ones_like(variables.dld_x_det, dtype=bool)
        if range_detx and range_dety:
            mask_det_x = (variables.dld_x_det < range_detx[1]) & (variables.dld_x_det > range_detx[0])
            mask_det_y = (variables.dld_y_det < range_dety[1]) & (variables.dld_y_det > range_dety[0])
            mask_det = mask_det_x & mask_det_y
        else:
            mask_det = np.ones(len(variables.dld_x_det), dtype=bool)
        if range_mc:
            mask_mc = (variables.mc_uc < range_mc[1]) & (variables.mc_uc > range_mc[0])
        else:
            mask_mc = np.ones(len(variables.mc), dtype=bool)
        if range_x and range_y and range_z:
            mask_x = (variables.x < range_x[1]) & (variables.x > range_x[0])
            mask_y = (variables.y < range_y[1]) & (variables.y > range_y[0])
            mask_z = (variables.z < range_z[1]) & (variables.z > range_z[0])
            mask_3d = mask_x & mask_y & mask_z
        else:
            mask_3d = np.ones(len(variables.x), dtype=bool)
        if range_vol:
            mask_vol = (variables.volume < range_vol[1]) & (variables.volume > range_vol[0])
        else:
            mask_vol = np.ones(len(variables.volume), dtype=bool)
        mask = mask_sequence & mask_det & mask_mc & mask_3d & mask_vol
        print('The number of data sequence:', len(mask_sequence[mask_sequence == True]))
        print('The number of data mc:', len(mask_mc[mask_mc == True]))
        print('The number of data det:', len(mask_det[mask_det == True]))
        print('The number of data 3d:', len(mask_3d[mask_3d == True]))
        print('The number of data after cropping:', len(mask[mask == True]))
    else:
        mask = np.ones(len(variables.mc), dtype=bool)

    ions = variables.range_data['ion'].tolist()
    colors = variables.range_data['color'].tolist()
    mc_low = variables.range_data['mc_low'].tolist()
    mc_up = variables.range_data['mc_up'].tolist()

    if isinstance(element_percentage, list):
        pass
    else:
        print('element_percentage should be a list')

    for index, elemen in enumerate(ions):
        mask_spacial = (variables.mc > mc_low[index]) & (variables.mc < mc_up[index])
        mask_s = mask & mask_spacial
        size = int(len(mask_s[mask_s == True]) * float(element_percentage[index]))
        # Find indices where the original mask is True
        true_indices = np.where(mask_s)[0]
        # Randomly choose 100 indices from the true indices
        random_true_indices = np.random.choice(true_indices, size=size, replace=False)
        # Create a new mask with the same length as the original, initialized with False
        new_mask = np.full(len(variables.mc), False)

        # Set the selected indices to True in the new mask
        new_mask[random_true_indices] = True
        # Apply the new mask to the original mask
        new_mask = mask_s & new_mask
        if ions[index] == 'unranged':
            name_element = 'unranged'
        else:
            name_element = '%s' % ions[index]

        ax.scatter(variables.dld_x_det[new_mask] * 10, variables.dld_y_det[new_mask] * 10, s=2, label=name_element,
                   color=colors[index], alpha=0.1)

    ax.set_xlabel("det_x (cm)", color="red", fontsize=10)
    ax.set_ylabel("det_y (cm)", color="red", fontsize=10)
    plt.title("Detector Heatmap")
    if len(variables.range_data) > 1:
        plt.legend(loc='upper right')

    if save:
        # Enable rendering for text elements
        rcParams['svg.fonttype'] = 'none'
        plt.savefig(variables.result_path + figure_name + "heatmap.png", format="png", dpi=600)
        plt.savefig(variables.result_path + figure_name + "heatmap.svg", format="svg", dpi=600)
    plt.show()


def reconstruction_2d_histogram(variables, x, y, bins, percentage, range_sequence=[], range_mc=[], range_detx=[],
                                range_dety=[], range_x=[], range_y=[], range_z=[], range_vol=[], xlabel='X-axis',
                                ylabel='Y-axis', save=False, figure_name=None, figure_size=None):
    """
    Generate a 2D histogram based on the provided data.

    Args:
        variables (object): The variables object.
        x (array): The x-axis data.
        y (array): The y-axis data.
        bins (int or tuple): The number of bins.
        percentage (float): percent of data to be plotted.
        range_sequence: Range of sequence
        range_mc: Range of mc
        range_detx: Range of detx
        range_dety: Range of dety
        range_x: Range of x-axis
        range_y: Range of y-axis
        range_z: Range of z-axis
        range_vol: Range of volume
        xlabel (str): The label of the x-axis.
        ylabel (str): The label of the y-axis.
        save (bool): True to save the plot, False to display it.
        figure_name (str): The name of the figure.
        figure_size (tuple): The size of the figure.

    Returns:
        None
    """
    if range_sequence or range_mc or range_detx or range_dety or range_x or range_y or range_z:
        if range_sequence:
            mask_sequence = np.zeros_like(variables.dld_x_det, dtype=bool)
            mask_sequence[range_sequence[0]:range_sequence[1]] = True
        else:
            mask_sequence = np.ones_like(variables.dld_x_det, dtype=bool)
        if range_detx and range_dety:
            mask_det_x = (variables.dld_x_det < range_detx[1]) & (variables.dld_x_det > range_detx[0])
            mask_det_y = (variables.dld_y_det < range_dety[1]) & (variables.dld_y_det > range_dety[0])
            mask_det = mask_det_x & mask_det_y
        else:
            mask_det = np.ones(len(variables.dld_x_det), dtype=bool)
        if range_mc:
            mask_mc = (variables.mc_uc < range_mc[1]) & (variables.mc_uc > range_mc[0])
        else:
            mask_mc = np.ones(len(variables.mc), dtype=bool)
        if range_x and range_y and range_z:
            mask_x = (variables.x < range_x[1]) & (variables.x > range_x[0])
            mask_y = (variables.y < range_y[1]) & (variables.y > range_y[0])
            mask_z = (variables.z < range_z[1]) & (variables.z > range_z[0])
            mask_3d = mask_x & mask_y & mask_z
        else:
            mask_3d = np.ones(len(variables.x), dtype=bool)
        if range_vol:
            mask_vol = (variables.volume < range_vol[1]) & (variables.volume > range_vol[0])
        else:
            mask_vol = np.ones(len(variables.volume), dtype=bool)
        mask = mask_sequence & mask_det & mask_mc & mask_3d & mask_vol
        print('The number of data sequence:', len(mask_sequence[mask_sequence == True]))
        print('The number of data mc:', len(mask_mc[mask_mc == True]))
        print('The number of data det:', len(mask_det[mask_det == True]))
        print('The number of data 3d:', len(mask_3d[mask_3d == True]))
        print('The number of data after cropping:', len(mask[mask == True]))
    else:
        mask = np.ones(len(variables.mc_uc), dtype=bool)

    x = x[mask]
    y = y[mask]

    num_elements_to_select = int(len(x) * percentage)
    # Randomly select elements
    indices = np.random.choice(len(x), num_elements_to_select, replace=False)
    x = x[indices]
    y = y[indices]
    # Check if the bin is a tuple
    if isinstance(bins, tuple):
        pass
    else:
        x_edges = np.arange(x.min(), x.max() + bins, bins)
        y_edges = np.arange(y.min(), y.max() + bins, bins)
        bins = [x_edges, y_edges]

    fig, ax = plt.subplots(figsize=figure_size)

    hist, xedges, yedges, _ = plt.hist2d(x, y, bins=bins, cmap='viridis')

    # # Add a colorbar
    cmap = copy(plt.cm.plasma)
    cmap.set_bad(cmap(0))

    pcm = ax.pcolormesh(xedges, yedges, hist.T, cmap=cmap, norm=colors.LogNorm(), rasterized=True)
    cbar = fig.colorbar(pcm, ax=ax, pad=0)
    cbar.set_label('Event Counts', fontsize=10)

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save:
        # Enable rendering for text elements
        rcParams['svg.fonttype'] = 'none'
        plt.savefig(variables.result_path + figure_name + ".png", format="png", dpi=600)
        plt.savefig(variables.result_path + figure_name + ".svg", format="svg", dpi=600)
    # Show the plot
    plt.show()


def detector_animation(variables, points_per_frame, ranged, selected_area_specially, selected_area_temporally,
                       figure_name, figure_sie, save):
    """
    Generate a animated heatmap based on the provided data.

    Args:
        variables (object): The variables object.
        points_per_frame (int): The number of points per frame.
        ranged (bool): True if the data is ranged, False otherwise.
        selected_area_specially (bool): True if a specific area is selected, False otherwise.
        selected_area_temporally (bool): True if a specific area is selected, False otherwise.
        figure_name (str): The name of the figure.
        figure_sie: The size of the figure.
        save (bool): True to save the plot, False to display it.

    Returns:
        None
    """
    if selected_area_specially:
        mask_spacial = (variables.x >= variables.selected_x1) & (variables.x <= variables.selected_x2) & \
                       (variables.y >= variables.selected_y1) & (variables.y <= variables.selected_y2) & \
                       (variables.z >= variables.selected_z1) & (variables.z <= variables.selected_z2)
    elif selected_area_temporally:
        mask_spacial = np.logical_and((variables.mc_calib > variables.selected_x1),
                                      (variables.mc_calib < variables.selected_x2))
    elif selected_area_specially and selected_area_temporally:
        mask_temporally = np.logical_and((variables.mc_calib > variables.selected_x1),
                                         (variables.mc_calib < variables.selected_x2))
        mask_specially = (variables.x >= variables.selected_x1) & (variables.x <= variables.selected_x2) & \
                         (variables.y >= variables.selected_y1) & (variables.y <= variables.selected_y2) & \
                         (variables.z >= variables.selected_z1) & (variables.z <= variables.selected_z2)
        mask_spacial = mask_specially & mask_temporally
    else:
        mask_spacial = np.ones(len(variables.mc), dtype=bool)

    if ranged == True:
        ions = variables.range_data['ion'].tolist()
        colors = variables.range_data['color'].tolist()
        mc_low = variables.range_data['mc_low'].tolist()
        mc_up = variables.range_data['mc_up'].tolist()
    else:
        ions = ['unranged']
        colors = ['black']
        mc_low = [0]
        mc_up = [400]

    x_data = variables.dld_x_det[mask_spacial]
    y_data = variables.dld_y_det[mask_spacial]

    # Define the number of points per frame
    points_per_frame = 5000

    # Calculate the total number of frames
    total_frames = len(x_data) // points_per_frame

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figure_sie)

    # Function to update the scatter plot for each frame
    def update(frame):
        ax.clear()
        start_idx = frame * points_per_frame
        end_idx = (frame + 1) * points_per_frame
        mc = variables.mc[start_idx:end_idx]
        for index, elemen in enumerate(ions):
            mask = (mc > mc_low[index]) & (mc < mc_up[index])
            mask = mask & mask_spacial[start_idx:end_idx]
            x = x_data[start_idx:end_idx][mask]
            y = y_data[start_idx:end_idx][mask]
            if ions[index] == 'unranged':
                name_element = 'unranged'
            else:
                name_element = '%s' % ions[index]
            ax.scatter(x, y, s=2, label=name_element,
                       color=colors[index], alpha=0.1)
            ax.set_title(f'Ion index: {start_idx} to {end_idx}')
            ax.set_xlabel("det_x (cm)", color="red", fontsize=10)
            ax.set_ylabel("det_y (cm)", color="red", fontsize=10)

    # Create an animation
    animation = FuncAnimation(fig, update, frames=total_frames, interval=500)

    # Convert the animation to HTML
    variables.animation_detector_html = animation.to_jshtml()

    if save:
        # Enable rendering for text elements
        rcParams['svg.fonttype'] = 'none'
        animation.save(variables.result_path + figure_name + ".gif", writer='imagemagick')
    plt.close()
def x_y_z_calculation_and_plot(variables, element_percentage, kf, det_eff, icf, field_evap,
                               avg_dens, flight_path_length, rotary_fig_save, mode, opacity, figname, save,
                               colab=False):
    """
    Calculate the x, y, z coordinates of the atoms and plot them.

        Args:
            variables (object): The variables object.
            element_percentage (str): Element percentage information.
            kf (float): The kinetic energy of the ions.
            det_eff (float): The detector efficiency.
            icf (float): The image compression factor.
            field_evap (float): The field evaporation efficiency.
            avg_dens (float): The average density of the atoms.
            flight_path_length (float): The flight path length.
            pulse_mode (str):
            rotary_fig_save (bool): True to save the rotary plot, False to display it.
            mode (str): The reconstruction mode.
            opacity (float): The opacity of the markers.
            figname (str): The name of the figure.
            save (bool): True to save the plot, False to display it.
            colab (bool): True if the code is running in Google Colab, False otherwise.

        Returns:
            None

    """
    if variables.pulse_mode == 'laser':
        dld_Voltage = variables.dld_high_voltage
    elif variables.pulse_mode == 'voltage':
        dld_Voltage = variables.dld_high_voltage + variables.dld_pulse_v
    dld_x = variables.dld_x_det
    dld_y = variables.dld_y_det
    if mode == 'Gault':
        px, py, pz = atom_probe_recons_from_detector_Gault_et_al(dld_x, dld_y, dld_Voltage,
                                                                 flight_path_length, kf, det_eff, icf,
                                                                 field_evap, avg_dens)
    elif mode == 'Bas':
        px, py, pz = atom_probe_recons_Bas_et_al(dld_x, dld_y, dld_Voltage, flight_path_length, kf, det_eff,
                                                 icf, field_evap, avg_dens)
    variables.x = px
    variables.y = py
    variables.z = pz
    reconstruction_plot(variables, element_percentage, opacity, rotary_fig_save, figname, save, colab=colab)
