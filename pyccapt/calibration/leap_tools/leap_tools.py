import re
import struct
import sys
from enum import Enum
from typing import Union, Tuple, Any
from warnings import warn

import matplotlib.colors as cols
import numpy as np
import pandas as pd
from vispy import app, scene


def read_pos(file_path):
    """
    Loads an APT .pos file as a pandas DataFrame.

    Columns:
        x: Reconstructed x position
        y: Reconstructed y position
        z: Reconstructed z position
        Da: Mass/charge ratio of ion
    """
    with open(file_path, 'rb') as file:
        data = file.read()
        n = len(data) // 4
        d = struct.unpack('>' + 'f' * n, data)
    pos = pd.DataFrame({
        'x (nm)': d[0::4],
        'y (nm)': d[1::4],
        'z (nm)': d[2::4],
        'm/n (Da)': d[3::4]
    })
    return pos


def read_epos(file_path):
    """
    Loads an APT .epos file as a pandas DataFrame.

    Columns:
        x: Reconstructed x position
        y: Reconstructed y position
        z: Reconstructed z position
        Da: Mass/charge ratio of ion
        ns: Ion Time Of Flight
        DC_kV: Potential
        pulse_kV: Size of voltage pulse (voltage pulsing mode only)
        det_x: Detector x position
        det_y: Detector y position
        pslep: Pulses since last event pulse (i.e. ionisation rate)
        ipp: Ions per pulse (multihits)
    """
    with open(file_path, 'rb') as file:
        data = file.read()

    n = len(data) // 4
    rs = n // 11
    d = struct.unpack('>' + 'fffffffffII' * rs, data)
    epos = pd.DataFrame({
        'x (nm)': d[0::11],
        'y (nm)': d[1::11],
        'z (nm)': d[2::11],
        'm/n (Da)': d[3::11],
        'TOF (ns)': d[4::11],
        'HV_DC (V)': d[5::11],
        'pulse (V)': d[6::11],
        'det_x (mm)': d[7::11],
        'det_y (mm)': d[8::11],
        'pslep': d[9::11],
        'ipp': d[10::11]
    })
    return epos


def read_rrng(file_path):
    """
    Loads a .rrng file produced by IVAS. Returns two DataFrames of 'ions' and 'ranges'.

    Parameters:
    - file_path (str): The path to the .rrng file.

    Returns:
    - ions (DataFrame): A DataFrame containing ion data with columns 'number' and 'name'.
    - rrngs (DataFrame): A DataFrame containing range data with columns 'number', 'lower', 'upper', 'vol', 'comp', and 'colour'.
    """

    # Read the file and store its contents as a list of lines
    rf = open(file_path, 'r').readlines()

    # Define the regular expression pattern to extract ion and range data
    patterns = re.compile(
        r'Ion([0-9]+)=([A-Za-z0-9]+).*|Range([0-9]+)=(\d+.\d+) +(\d+.\d+) +Vol:(\d+.\d+) +([A-Za-z:0-9 ]+) +Color:([A-Z0-9]{6})')

    # Initialize empty lists to store ion and range data
    ions = []
    rrngs = []

    # Iterate over each line in the file
    for line in rf:
        # Search for matches using the regular expression pattern
        m = patterns.search(line)
        if m:
            # If match groups contain ion data, append to ions list
            if m.groups()[0] is not None:
                ions.append(m.groups()[:2])
            # If match groups contain range data, append to rrngs list
            else:
                rrngs.append(m.groups()[2:])

    mc_low = [float(i[1].replace(',', '.')) for i in rrngs]
    mc_up = [float(i[2].replace(',', '.')) for i in rrngs]
    mc = [(float(i[1].replace(',', '.')) + float(i[2].replace(',', '.'))) / 2 for i in rrngs]
    elements = [i[4] for i in rrngs]
    colors = [i[5] for i in rrngs]
    charge = [1] * len(rrngs)
    # Output lists
    complex = []
    element_list = []
    ion_list = []
    # Process each item in the input list
    for item in elements:
        # Split by space if there are multiple elements (e.g., 'Mo:1 O:3')
        parts = item.split()

        # Initialize lists for complexity and elements
        complexities = []
        elements_s = []
        for part in parts:
            # Split by colon to separate element and complexity
            element, complexity = part.split(':')
            if element == 'Name':
                element = 'unranged'
                complexity = 0
            # Append element and complexity
            elements_s.append(element)
            complexities.append(int(complexity))
        # Append the result for each item
        complex.append(complexities)
        element_list.append(elements_s)

    # make isotope list of list base on element list
    isotope = []

    for i in range(len(element_list)):
        isotope_s = []
        for j in range(len(element_list[i])):
            formula = r'$'
            formula += '{}^'
            formula += '{%s}' % 1
            formula += '%s' % element_list[i][0]
            if complex[i][j] > 1:
                formula += '_{%s}' % complex[i][j]
            isotope_s.append(1)
        if charge[i] > 1:
            formula += '^{%s+}$' % charge[i]
        else:
            formula += '^{+}$'
        isotope.append(isotope_s)
        ion_list.append(formula)

    name = []
    for i in range(len(element_list)):
        name.append(".".join(f"{element_list[i][j]}{complex[i][j]}" for j in range(len(element_list[i]))))

    # Return the pyccapt_ranges DataFrame
    range_data = pd.DataFrame({'name': name, 'ion': ion_list, 'mass': mc, 'mc': mc, 'mc_low': mc_low,
                                    'mc_up': mc_up, 'color': colors, 'element': element_list,
                                    'complex': complex, 'isotope': isotope, 'charge': charge})
    return range_data


def write_rrng(file_path, ions, rrngs):
    """
    Writes two DataFrames of 'ions' and 'ranges' to a .rrng file in IVAS format.

    Parameters:
    - file_path (str): The path to the .rrng file to be created.
    - ions (DataFrame): A DataFrame containing ion data with columns 'number' and 'name'.
    - rrngs (DataFrame): A DataFrame containing range data with columns 'number', 'lower', 'upper', 'vol', 'comp',
      and 'color'.

    Returns:
    None
    """
    with open(file_path, 'w') as f:
        # Write ion data
        f.write('[Ions]\n')
        for index, row in ions.iterrows():
            ion_line = f'Ion{index}={row["name"]}\n'
            f.write(ion_line)

        # Write range data
        f.write('[Ranges]\n')
        for index, row in rrngs.iterrows():
            range_line = f'Range{index}={row["lower"]:.2f} {row["upper"]:.2f} Vol:{row["vol"]:.2f} {row["comp"]} Color:{row["color"]}\n'
            f.write(range_line)


def label_ions(pos, rrngs):
    """
    Labels ions in a .pos or .epos DataFrame (anything with a 'Da' column) with composition and color,
    based on an imported .rrng file.

    Parameters:
    - pos (DataFrame): A DataFrame containing ion positions, with a 'Da' column.
    - rrngs (DataFrame): A DataFrame containing range data imported from a .rrng file.

    Returns:
    - pos (DataFrame): The modified DataFrame with added 'comp' and 'colour' columns.
    """

    # Initialize 'comp' and 'colour' columns in the DataFrame pos
    pos['comp'] = ''
    pos['colour'] = '#FFFFFF'

    # Iterate over each row in the DataFrame rrngs
    for n, r in rrngs.iterrows():
        # Assign composition and color values to matching ion positions in pos DataFrame
        pos.loc[(pos['Da'] >= r.lower) & (pos['Da'] <= r.upper), ['comp', 'colour']] = [r['comp'], '#' + r['colour']]

    # Return the modified pos DataFrame with labeled ions
    return pos


def deconvolve(lpos):
    """
    Takes a composition-labelled pos file and deconvolves the complex ions.
    Produces a DataFrame of the same input format with the extra columns:
    'element': element name
    'n': stoichiometry
    For complex ions, the location of the different components is not altered - i.e. xyz position will be the same
    for several elements.

    Parameters:
    - lpos (DataFrame): A composition-labelled pos file DataFrame.

    Returns:
    - out (DataFrame): A deconvolved DataFrame with additional 'element' and 'n' columns.
    """

    # Initialize an empty list to store the deconvolved data
    out = []

    # Define the regular expression pattern to extract element and stoichiometry information
    pattern = re.compile(r'([A-Za-z]+):([0-9]+)')

    # Group the input DataFrame 'lpos' based on the 'comp' column
    for g, d in lpos.groupby('comp'):
        if g != '':
            # Iterate over the elements in the 'comp' column
            for i in range(len(g.split(' '))):
                # Create a copy of the grouped DataFrame 'd'
                tmp = d.copy()
                # Extract the element and stoichiometry values using the regular expression pattern
                cn = pattern.search(g.split(' ')[i]).groups()
                # Add 'element' and 'n' columns to the copy of DataFrame 'tmp'
                tmp['element'] = cn[0]
                tmp['n'] = cn[1]
                # Append the modified DataFrame 'tmp' to the output list
                out.append(tmp.copy())

    # Concatenate the DataFrame in the output list to create the final deconvolved DataFrame
    return pd.concat(out)


def volvis(pos, size=2, alpha=1):
    """
    Displays a 3D point cloud in an OpenGL viewer window. If points are not labelled with colors,
    point brightness is determined by Da values (higher = whiter).

    Parameters:
    - pos (DataFrame): A DataFrame containing 3D point cloud data.
    - size (int): The size of the markers representing the points. Default is 2.
    - alpha (float): The transparency of the markers. Default is 1.

    Returns:
    - None
    """

    # Create an OpenGL viewer window
    canvas = scene.SceneCanvas('APT Volume', keys='interactive')
    view = canvas.central_widget.add_view()
    view.camera = scene.TurntableCamera(up='z')

    # Extract the position data from the 'pos' DataFrame
    cpos = pos[['x (nm)', 'y (nm)', 'z (nm)']].values

    # Check if the 'colour' column is present in the 'pos' DataFrame
    if 'colour' in pos.columns:
        # Extract colors from the 'colour' column
        colours = np.asarray(list(pos['colour'].apply(cols.hex2color)))
    else:
        # Calculate brightness based on Da values
        Dapc = pos['m/n (Da)'].values / pos['m/n (Da)'].max()
        colours = np.array(zip(Dapc, Dapc, Dapc))

    # Adjust colors based on transparency (alpha value)
    if alpha != 1:
        colours = np.hstack([colours, np.array([0.5] * len(colours))[..., None]])

    # Create and configure markers for the point cloud
    p1 = scene.visuals.Markers()
    p1.set_data(cpos, face_color=colours, edge_width=0, size=size)

    # Add the markers to the viewer
    view.add(p1)

    # Create arrays to store ion labels and corresponding colors
    ions = []
    cs = []

    # Group the 'pos' DataFrame by color
    for g, d in pos.groupby('colour'):
        # Remove ':' and whitespaces from the 'comp' column values
        ions.append(re.sub(r':1?|\s?', '', d['comp'].iloc[0]))
        cs.append(cols.hex2color(g))

    ions = np.array(ions)
    cs = np.asarray(cs)

    # Create positions and text for the legend
    pts = np.array([[20] * len(ions), np.linspace(20, 20 * len(ions), len(ions))]).T
    tpts = np.array([[30] * len(ions), np.linspace(20, 20 * len(ions), len(ions))]).T

    # Create a legend box
    legb = scene.widgets.ViewBox(parent=view, border_color='red', bgcolor='k')
    legb.pos = 0, 0
    legb.size = 100, 20 * len(ions) + 20

    # Create markers for the legend
    leg = scene.visuals.Markers()
    leg.set_data(pts, face_color=cs)
    legb.add(leg)

    # Add text to the legend
    legt = scene.visuals.Text(text=ions, pos=tpts, color='white', anchor_x='left', anchor_y='center', font_size=10)
    legb.add(legt)

    # Display the canvas
    canvas.show()

    # Run the application event loop if not running interactively
    if sys.flags.interactive == 0:
        app.run()





class RelationKind(Enum):
    UNSPECIFIED = 0
    SINGLE = 1
    INDEXED = (2,)
    INDEPENDENT = 3
    MULTIPLE = 4

class DataCategory(Enum):
    UNSPECIFIED = 0
    CONSTANT = 1
    VARIABLE = 2
    INDEXED_VARIABLE = 3

class DataFormat(Enum):
    UNSPECIFIED = 0
    INTEGER = 1
    UNSIGNED_INT = 2
    DECIMAL = 3
    TEXT = 4
    CUSTOM = 5

class ByteFormat(Enum):
    INT_32 = 4
    INT_64 = 8
    CHAR = 1
    WIDE_CHAR = 2
    TIME_STAMP = 8

def read_apt(file_path: str, debug: bool = False) -> pd.DataFrame:
    """
    Load data from an APT file into a pandas DataFrame.

    Args:
        file_path (str): The path to the APT file.
        debug (bool): If True, print detailed information during loading.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """

    def map_data_type(data_format: DataFormat, bit_size: int):
        """
        Convert a data format and size to the corresponding numpy data type.
        """
        int_types = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}
        uint_types = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
        float_types = {32: np.float32, 64: np.float64}

        if data_format == DataFormat.INTEGER:
            return int_types[bit_size]
        elif data_format == DataFormat.UNSIGNED_INT:
            return uint_types[bit_size]
        elif data_format == DataFormat.DECIMAL:
            return float_types[bit_size]
        else:
            raise ValueError(f"Unsupported data format: {data_format}")

    format_map = {
        ByteFormat.INT_32: "i",
        ByteFormat.INT_64: "q",
        ByteFormat.CHAR: "c",
        ByteFormat.TIME_STAMP: "Q",
        ByteFormat.WIDE_CHAR: "c",
    }

    type_constructors = {
        ByteFormat.INT_32: int,
        ByteFormat.INT_64: int,
        ByteFormat.CHAR: lambda x: x.decode("utf-8"),
        ByteFormat.WIDE_CHAR: lambda x: x.decode("utf-16"),
        ByteFormat.TIME_STAMP: int,
    }

    with open(file_path, "rb") as file:

        def extract_data(data_type: ByteFormat, num_items: int = 1, position: Union[None, int] = None) -> Union[Tuple[Any], Any]:
            if isinstance(position, int):
                file.seek(position)

            fmt = format_map[data_type] * num_items
            constructor = type_constructors[data_type]
            data_size = data_type.value

            if data_type in (ByteFormat.WIDE_CHAR, ByteFormat.CHAR):
                return constructor(file.read(data_size * num_items)).replace("\x00", "")
            else:
                result = struct.unpack("<" + fmt, file.read(data_size * num_items))

            if len(result) == 1:
                return constructor(result[0])
            else:
                return tuple(constructor(i) for i in result)

        signature = extract_data(ByteFormat.CHAR, 4)

        header_size = extract_data(ByteFormat.INT_32)
        header_version = extract_data(ByteFormat.INT_32)
        file_name = extract_data(ByteFormat.WIDE_CHAR, 256)
        creation_time = extract_data(ByteFormat.TIME_STAMP)
        ion_count = extract_data(ByteFormat.INT_64)

        if debug:
            print(f"\nLoading header from {file_path}")
            print(f"\tSignature: {signature}")
            print(f"\tHeader Size: {header_size}")
            print(f"\tHeader Version: {header_version}")
            print(f"\tFile Name: {file_name}")
            print(f"\tCreation Time: {creation_time}")
            print(f"\tIon Count: {ion_count}")

        current_position = header_size
        data_sections = {}

        while True:
            section_signature = extract_data(ByteFormat.CHAR, 4, current_position)

            if section_signature == "":
                break

            skip_section = False

            section_header_size = extract_data(ByteFormat.INT_32)
            section_header_version = extract_data(ByteFormat.INT_32)
            section_name = extract_data(ByteFormat.WIDE_CHAR, 32)
            section_version = extract_data(ByteFormat.INT_32)

            section_relation = RelationKind(extract_data(ByteFormat.INT_32))
            if section_relation != RelationKind.SINGLE:
                warn(f'Unsupported relation type: {section_relation}, section "{section_name}" will be skipped')
                skip_section = True

            section_category = DataCategory(extract_data(ByteFormat.INT_32))
            if section_category != DataCategory.CONSTANT:
                warn(f'Unsupported data category: {section_category}, section "{section_name}" will be skipped')
                skip_section = True

            section_format = DataFormat(extract_data(ByteFormat.INT_32))
            if section_format in (DataFormat.UNSPECIFIED, DataFormat.CUSTOM, DataFormat.TEXT):
                warn(f'Unsupported data format: {section_format}, section "{section_name}" will be skipped')
                skip_section = True

            section_bit_size = extract_data(ByteFormat.INT_32)
            section_record_size = extract_data(ByteFormat.INT_32)
            section_unit = extract_data(ByteFormat.WIDE_CHAR, 16)
            section_record_count = extract_data(ByteFormat.INT_64)
            section_byte_count = extract_data(ByteFormat.INT_64)

            if debug:
                print("\nLoading new section")
                print(f"\tSection Signature: {section_signature}")
                print(f"\tSection Header Size: {section_header_size}")
                print(f"\tSection Header Version: {section_header_version}")
                print(f"\tSection Name: {section_name}")
                print(f"\tSection Version: {section_version}")
                print(f"\tSection Relation: {section_relation}")
                print(f"\tSection Category: {section_category}")
                print(f"\tSection Format: {section_format}")
                print(f"\tSection Bit Size: {section_bit_size}")
                print(f"\tSection Record Size: {section_record_size}")
                print(f"\tSection Unit: {section_unit}")
                print(f"\tSection Record Count: {section_record_count}")
                print(f"\tSection Byte Count: {section_byte_count}")

            if not skip_section:
                num_columns = int(section_record_size / (section_bit_size / 8))
                num_records = int(section_record_count)
                total_items = num_records * num_columns
                section_data = np.fromfile(
                    file_path,
                    map_data_type(section_format, section_bit_size),
                    total_items,
                    offset=current_position + section_header_size,
                )
                if num_columns > 1:
                    data_sections[section_name] = section_data.reshape(num_records, num_columns)
                else:
                    data_sections[section_name] = section_data

            current_position = current_position + section_byte_count + section_header_size

    has_mass = "Mass" in data_sections.keys()
    has_position = "Position" in data_sections.keys()

    if not has_mass:
        raise AttributeError("APT file must include a mass section")
    elif not has_position:
        raise AttributeError("APT file must include a position section")

    if "Detector Coordinates" in data_sections.keys():
        temp = data_sections.pop("Detector Coordinates")
        if "XDet_mm" not in data_sections.keys():
            data_sections["XDet_mm"] = temp[:, 0]
        if "YDet_mm" not in data_sections.keys():
            data_sections["YDet_mm"] = temp[:, 1]

    if "Position" in data_sections.keys():
        temp = data_sections.pop("Position")
        if "x" not in data_sections.keys():
            data_sections["x"] = temp[:, 0]
        if "y" not in data_sections.keys():
            data_sections["y"] = temp[:, 1]
        if "z" not in data_sections.keys():
            data_sections["z"] = -1 * temp[:, 2]

    if debug:
        for section in data_sections.keys():
            print(f"Section: {section} - {data_sections[section].shape} - {data_sections[section].dtype} - {data_sections[section]}")
    df = pd.DataFrame(data_sections)

    return df