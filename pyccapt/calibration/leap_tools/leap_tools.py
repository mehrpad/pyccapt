import re
import struct
import sys
import os
from enum import Enum
from typing import Union, Tuple, Any
from warnings import warn

import matplotlib.colors as cols
import numpy as np
import pandas as pd
from vispy import app, scene


_RRNG_PATTERN = re.compile(
    r"Ion(?P<ion_number>\d+)=(?P<ion_name>.+)"
    r"|Range(?P<range_number>\d+)="
    r"(?P<lower>[+-]?\d+(?:[.,]\d+)?)\s+"
    r"(?P<upper>[+-]?\d+(?:[.,]\d+)?)\s+"
    r"Vol:(?P<vol>[+-]?\d+(?:[.,]\d+)?)\s+"
    r"(?P<comp>.+?)\s+Color:(?P<colour>[A-Fa-f0-9]{6})$"
)
_RNG_COUNTS_PATTERN = re.compile(r"^\d+\s+\d+$")
_FORMULA_TOKEN_PATTERN = re.compile(r"([A-Z][a-z]*)(\d*)")
_SPECIES_TOKEN_PATTERN = re.compile(r"(?:(?P<isotope>\d+))?(?P<element>[A-Z][a-z]*)(?P<count>\d*)")


def _parse_rrng_float(value: str) -> float:
    """Parse `.rrng` numeric fields while accepting comma decimals."""
    return float(str(value).strip().replace(",", "."))


def _parse_rrng_species(species: str):
    """Parse compact species names like `Al2O3` or isotope-prefixed `27CrMo`."""
    species = str(species).strip()
    if not species or species.lower() == "unranged":
        return ["unranged"], [0], [0]

    matches = list(_SPECIES_TOKEN_PATTERN.finditer(species))
    if not matches or "".join(match.group(0) for match in matches) != species:
        return [species], [1], [0]

    elements = []
    multiplicities = []
    isotopes = []
    for match in matches:
        elements.append(match.group("element"))
        multiplicities.append(int(match.group("count") or "1"))
        isotopes.append(int(match.group("isotope") or "0"))

    return elements, multiplicities, isotopes


def _parse_rrng_element_token(token: str):
    """Parse one element token such as `Fe` or `27Cr`."""
    token = str(token).strip()
    match = _SPECIES_TOKEN_PATTERN.fullmatch(token)
    if not match:
        return token, 0
    return match.group("element"), int(match.group("isotope") or "0")


def _parse_rrng_composition(composition: str, explicit_name: str = ""):
    """Parse the IVAS composition field into PyCCAPT element metadata."""
    tokens = [token for token in str(composition).split() if token]
    if not tokens:
        return _parse_rrng_species(explicit_name or "unranged")

    elements = []
    multiplicities = []
    isotopes = []

    for token in tokens:
        if ":" in token:
            element_token, value = token.split(":", 1)
            element_token = element_token.strip()
            value = value.strip()

            if element_token.lower() == "name":
                species_name = explicit_name or ("unranged" if value in {"0", "0.0"} else value)
                species_elements, species_multiplicities, species_isotopes = _parse_rrng_species(species_name)
                elements.extend(species_elements)
                multiplicities.extend(species_multiplicities)
                isotopes.extend(species_isotopes)
                continue

            try:
                multiplicity_value = int(float(value))
            except ValueError:
                species_name = explicit_name or value or element_token
                species_elements, species_multiplicities, species_isotopes = _parse_rrng_species(species_name)
                elements.extend(species_elements)
                multiplicities.extend(species_multiplicities)
                isotopes.extend(species_isotopes)
                continue

            element_name, isotope_value = _parse_rrng_element_token(element_token)
            elements.append(element_name)
            multiplicities.append(multiplicity_value)
            isotopes.append(isotope_value)
            continue

        species_elements, species_multiplicities, species_isotopes = _parse_rrng_species(token)
        elements.extend(species_elements)
        multiplicities.extend(species_multiplicities)
        isotopes.extend(species_isotopes)

    if not elements:
        return _parse_rrng_species(explicit_name or "unranged")

    return elements, multiplicities, isotopes


def _build_rrng_formula(elements, multiplicities, charge: int) -> str:
    """Build the lightweight LaTeX-style ion label used in the workflows."""
    if not elements or elements == ["unranged"]:
        return "unranged"

    parts = []
    for element, multiplicity in zip(elements, multiplicities):
        multiplicity_value = int(multiplicity)
        if multiplicity_value > 1:
            parts.append(f"{element}_{{{multiplicity_value}}}")
        else:
            parts.append(str(element))

    charge_value = max(int(charge), 1)
    charge_text = "+" if charge_value == 1 else f"{charge_value}+"
    return "$" + "".join(parts) + f"^{{{charge_text}}}$"


def _composition_to_name(elements, multiplicities) -> str:
    """Build a compact plain-text name from a composition list."""
    if not elements or elements == ["unranged"]:
        return "unranged"
    return "".join(
        f"{element}{'' if int(multiplicity) == 1 else int(multiplicity)}"
        for element, multiplicity in zip(elements, multiplicities)
    )


def _extract_explicit_rrng_name(composition: str, ion_name: str = "") -> str:
    """Extract a stable explicit name from RRNG metadata when available."""
    ion_name = str(ion_name).strip()
    if ion_name:
        return ion_name

    tokens = [token for token in str(composition).split() if token]
    if len(tokens) == 1 and ":" in tokens[0]:
        key, value = tokens[0].split(":", 1)
        if key.strip().lower() == "name":
            value = str(value).strip()
            if value in {"0", "0.0"}:
                return "unranged"
            return value
    return ""


def _range_tables_to_pyccapt_range(ions: pd.DataFrame, range_table: pd.DataFrame) -> pd.DataFrame:
    """Convert raw `.rrng` or legacy `.rng` tables into the normalized PyCCAPT schema."""
    if range_table.empty:
        return pd.DataFrame(
            columns=[
                "name",
                "ion",
                "mass",
                "mc",
                "mc_low",
                "mc_up",
                "color",
                "element",
                "complex",
                "isotope",
                "charge",
                "vol",
                "raw_comp",
                "ion_name",
            ]
        )

    names = []
    ion_labels = []
    element_lists = []
    complex_lists = []
    isotope_lists = []
    charges = []

    for _, row in range_table.iterrows():
        explicit_name = _extract_explicit_rrng_name(row["comp"], row.get("ion_name", ""))
        elements, multiplicities, isotopes = _parse_rrng_composition(row["comp"], explicit_name=explicit_name)
        charge = 1
        name = explicit_name or _composition_to_name(elements, multiplicities)
        names.append(name)
        ion_labels.append(_build_rrng_formula(elements, multiplicities, charge))
        element_lists.append(elements)
        complex_lists.append(multiplicities)
        isotope_lists.append(isotopes)
        charges.append(charge)

    mc_low = range_table["lower"].astype(float).to_numpy()
    mc_up = range_table["upper"].astype(float).to_numpy()
    pyccapt_range = pd.DataFrame(
        {
            "name": names,
            "ion": ion_labels,
            "mass": (mc_low + mc_up) / 2.0,
            "mc": (mc_low + mc_up) / 2.0,
            "mc_low": mc_low,
            "mc_up": mc_up,
            "color": ("#" + range_table["colour"].astype(str).str.upper().str.lstrip("#")).to_numpy(),
            "element": element_lists,
            "complex": complex_lists,
            "isotope": isotope_lists,
            "charge": np.asarray(charges, dtype=int),
            "vol": range_table["vol"].astype(float).to_numpy(),
            "raw_comp": range_table["comp"].astype(str).to_numpy(),
            "ion_name": np.asarray(names, dtype=object),
        }
    )
    pyccapt_range.attrs["range_ions"] = ions.copy()
    pyccapt_range.attrs["range_ranges"] = range_table.copy()
    return pyccapt_range


def parse_rrng(file_path):
    """
    Load a `.rrng` file produced by IVAS and return the raw ion and range tables.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            `(ions, rrngs)` in the IVAS schema.
    """
    ions = []
    rrngs = []

    with open(file_path, "r", encoding="utf-8") as range_file:
        for line in range_file:
            match = _RRNG_PATTERN.search(line.strip())
            if not match:
                continue

            groups = match.groupdict()
            if groups["ion_number"] is not None:
                ions.append((groups["ion_number"], groups["ion_name"].strip()))
                continue

            rrngs.append(
                (
                    groups["range_number"],
                    _parse_rrng_float(groups["lower"]),
                    _parse_rrng_float(groups["upper"]),
                    _parse_rrng_float(groups["vol"]),
                    groups["comp"].strip(),
                    groups["colour"].strip().upper(),
                )
            )

    ions_df = pd.DataFrame(ions, columns=["number", "name"])
    if not ions_df.empty:
        ions_df["number"] = ions_df["number"].astype(str)
        ions_df.set_index("number", inplace=True)

    rrngs_df = pd.DataFrame(rrngs, columns=["number", "lower", "upper", "vol", "comp", "colour"])
    if not rrngs_df.empty:
        rrngs_df["number"] = rrngs_df["number"].astype(str)
        rrngs_df.set_index("number", inplace=True)
        rrngs_df[["lower", "upper", "vol"]] = rrngs_df[["lower", "upper", "vol"]].astype(float)
        rrngs_df[["comp", "colour"]] = rrngs_df[["comp", "colour"]].astype(str)

    return ions_df, rrngs_df


def _rgb_float_to_hex(red: float, green: float, blue: float) -> str:
    """Convert 0-1 RGB triples from legacy `.rng` files to uppercase hex."""
    values = [int(round(max(0.0, min(1.0, float(value))) * 255.0)) for value in (red, green, blue)]
    return "".join(f"{value:02X}" for value in values)


def _formula_to_comp_string(formula: str) -> str:
    """Convert a compact formula like `Cr2O` to `Cr:2 O:1`."""
    pairs = []
    for element, count in _FORMULA_TOKEN_PATTERN.findall(str(formula).strip()):
        multiplicity = int(count) if count else 1
        pairs.append(f"{element}:{multiplicity}")
    return " ".join(pairs) if pairs else "Name:0"


def _average_hex_colours(colours: list[str]) -> str:
    """Blend multiple hex colours into one representative colour."""
    if not colours:
        return "FFFFFF"
    rgb = np.array(
        [[int(colour[i : i + 2], 16) for i in (0, 2, 4)] for colour in colours],
        dtype=float,
    )
    blended = np.round(rgb.mean(axis=0)).astype(int)
    return "".join(f"{value:02X}" for value in blended)


def _parse_rng_block(lines: list[str], start_index: int):
    """Parse one block of the legacy `.rng` text format."""
    counts = lines[start_index].split()
    nions = int(counts[0])
    nranges = int(counts[1])
    index = start_index + 1

    ions = []
    ion_colours: dict[str, str] = {}
    for ion_number in range(1, nions + 1):
        ion_name = lines[index].strip()
        index += 1
        colour_tokens = lines[index].split()
        index += 1
        colour = _rgb_float_to_hex(colour_tokens[1], colour_tokens[2], colour_tokens[3])
        ions.append((str(ion_number), ion_name, colour))
        ion_colours[ion_name] = colour

    separator_tokens = lines[index].split()
    index += 1
    headers = separator_tokens[1:] if separator_tokens else []

    ranges = []
    for range_number in range(1, nranges + 1):
        tokens = lines[index].split()
        index += 1
        if len(tokens) < 3 + len(headers):
            raise ValueError("Malformed legacy .rng range row")
        lower = float(tokens[1])
        upper = float(tokens[2])
        counts = [int(value) for value in tokens[3 : 3 + len(headers)]]
        composition = [(header, count) for header, count in zip(headers, counts) if count > 0]
        comp_string = " ".join(f"{name}:{count}" for name, count in composition) if composition else "Name:0"
        active_colours = [ion_colours.get(name, "FFFFFF") for name, count in composition if count > 0]
        ranges.append(
            {
                "number": str(range_number),
                "lower": lower,
                "upper": upper,
                "vol": np.nan,
                "comp": comp_string,
                "colour": _average_hex_colours(active_colours),
                "ion_name": _composition_to_name(
                    [name for name, _ in composition] or ["unranged"],
                    [count for _, count in composition] or [0],
                ),
            }
        )

    ions_df = pd.DataFrame(ions, columns=["number", "name", "colour"])
    if not ions_df.empty:
        ions_df.set_index("number", inplace=True)

    ranges_df = pd.DataFrame(ranges, columns=["number", "lower", "upper", "vol", "comp", "colour", "ion_name"])
    if not ranges_df.empty:
        ranges_df.set_index("number", inplace=True)

    return index, ions_df, ranges_df


def parse_rng(file_path):
    """
    Load a legacy IVAS/LEAP `.rng` file and return raw ion and range tables.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            `(ions, ranges)` in a normalized raw schema.
    """
    with open(file_path, "r", encoding="utf-8") as range_file:
        lines = [line.strip() for line in range_file if line.strip()]

    if not lines:
        raise ValueError("Empty .rng file")

    index, ions_df, ranges_df = _parse_rng_block(lines, 0)
    extension_map: dict[tuple[float, float], tuple[str, str]] = {}

    while index < len(lines):
        if lines[index].startswith("---"):
            index += 1
            if index >= len(lines):
                break
        if not _RNG_COUNTS_PATTERN.match(lines[index]):
            index += 1
            continue

        index, ext_ions_df, ext_ranges_df = _parse_rng_block(lines, index)
        for _, row in ext_ranges_df.iterrows():
            formula_name = str(row.get("ion_name", "")).strip()
            if not formula_name or formula_name == "unranged":
                continue
            key = (round(float(row["lower"]), 6), round(float(row["upper"]), 6))
            colour = str(row["colour"]).upper()
            matching_formula = None
            for _, ext_ion in ext_ions_df.iterrows():
                ext_name = str(ext_ion["name"]).strip()
                if _formula_to_comp_string(ext_name) == row["comp"]:
                    matching_formula = ext_name
                    colour = ext_ion["colour"]
                    break
            extension_map[key] = (matching_formula or formula_name, str(colour).upper())

    if not ranges_df.empty:
        for row_index, row in ranges_df.iterrows():
            key = (round(float(row["lower"]), 6), round(float(row["upper"]), 6))
            if key not in extension_map:
                continue
            formula_name, colour = extension_map[key]
            ranges_df.at[row_index, "ion_name"] = formula_name
            ranges_df.at[row_index, "colour"] = colour

    return ions_df, ranges_df


def read_pos(file_path):
    """
    Loads an APT .pos file as a pandas DataFrame.

    Columns:
        x: Reconstructed x position
        y: Reconstructed y position
        z: Reconstructed z position
        Da: Mass/charge ratio of ion

    Note:
        For low-memory environments use :func:`read_pos_lazy`, which returns a
        :class:`pyccapt.calibration.data_tools.lazy_io.LazyTable` view that
        reads columns on demand instead of materializing the full DataFrame.
    """
    from pyccapt.calibration.data_tools import lazy_io

    with lazy_io.open_pos(file_path) as table:
        return table.to_dataframe()


def read_pos_lazy(file_path):
    """Open an APT ``.pos`` file as a memory-mapped :class:`LazyTable`.

    Returns a column-oriented view that reads on demand. Use this in
    low-memory contexts; the caller is responsible for closing the table
    (preferred: ``with read_pos_lazy(path) as table: ...``).
    """
    from pyccapt.calibration.data_tools import lazy_io

    return lazy_io.open_pos(file_path)


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

    Note:
        For low-memory environments use :func:`read_epos_lazy`, which returns a
        :class:`pyccapt.calibration.data_tools.lazy_io.LazyTable` view that
        reads columns on demand. The eager :func:`read_epos` itself now reads
        the memmap field-by-field with a single byte-order conversion per
        column instead of the previous 11-fold ``np.asarray`` cascade.
    """
    from pyccapt.calibration.data_tools import lazy_io

    with lazy_io.open_epos(file_path) as table:
        return table.to_dataframe()


def read_epos_lazy(file_path):
    """Open an APT ``.epos`` file as a memory-mapped :class:`LazyTable`.

    Returns a column-oriented view that reads on demand. Suited for the
    low-memory case (e.g. correcting a 2 GB EPOS on an 8 GB machine). Use
    ``with read_epos_lazy(path) as table: ...`` so the memmap is released
    promptly on Windows.
    """
    from pyccapt.calibration.data_tools import lazy_io

    return lazy_io.open_epos(file_path)


def read_rrng(file_path, return_tables: bool = False):
    """
    Load a `.rrng` file produced by IVAS.

    Parameters:
    - file_path (str): The path to the `.rrng` file.
    - return_tables (bool): When `True`, return the raw `(ions, rrngs)`
      IVAS tables. Otherwise return the normalized PyCCAPT range dataframe.
    """
    ions, rrngs = parse_rrng(file_path)
    if return_tables:
        return ions, rrngs
    return _range_tables_to_pyccapt_range(ions, rrngs)


def read_rng(file_path, return_tables: bool = False):
    """
    Load a legacy `.rng` file produced by older IVAS/LEAP workflows.

    Parameters:
    - file_path (str): The path to the `.rng` file.
    - return_tables (bool): When `True`, return the raw `(ions, ranges)`
      tables. Otherwise return the normalized PyCCAPT range dataframe.
    """
    ions, ranges = parse_rng(file_path)
    if return_tables:
        return ions, ranges
    return _range_tables_to_pyccapt_range(ions, ranges)


def write_rrng(file_path, ions, rrngs):
    """
    Write ion and range DataFrames to an IVAS-style ``.rrng`` file.

    Parameters:
    - file_path (str): Destination path for the ``.rrng`` file.
    - ions (DataFrame): Ion table with at least the ``name`` column.
    - rrngs (DataFrame): Range table with ``lower``, ``upper``, ``vol``, ``comp``, and ``color`` columns.

    Returns:
    - None
    """
    with open(file_path, 'w') as f:
        # Write ion data
        f.write('[Ions]\n')
        for index, row in ions.iterrows():
            ion_line = f'Ion{index}={row["name"]}\n'
            f.write(ion_line)

        # Write range data
        f.write('[Ranges]\n')
        color_column = 'color' if 'color' in rrngs.columns else 'colour'
        for index, row in rrngs.iterrows():
            range_line = (
                f'Range{index}={row["lower"]:.2f} {row["upper"]:.2f} '
                f'Vol:{row["vol"]:.2f} {row["comp"]} Color:{str(row[color_column]).lstrip("#")}\n'
            )
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

    mass_column = None
    for candidate in ("Da", "m/n (Da)", "mc (Da)"):
        if candidate in pos.columns:
            mass_column = candidate
            break
    if mass_column is None:
        raise KeyError("The position dataframe must contain 'Da', 'm/n (Da)', or 'mc (Da)'")

    # Initialize 'comp' and 'colour' columns in the DataFrame pos
    pos['comp'] = ''
    pos['colour'] = '#FFFFFF'

    # Iterate over each row in the DataFrame rrngs
    for n, r in rrngs.iterrows():
        # Assign composition and color values to matching ion positions in pos DataFrame
        pos.loc[
            (pos[mass_column] >= r.lower) & (pos[mass_column] <= r.upper),
            ['comp', 'colour'],
        ] = [r['comp'], '#' + str(r['colour']).lstrip('#')]

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

        def extract_data(
            data_type: ByteFormat, num_items: int = 1, position: Union[None, int] = None
        ) -> Union[Tuple[Any], Any]:
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
        tipbox = None

        while True:
            section_signature = extract_data(ByteFormat.CHAR, 4, current_position)

            if section_signature == "":
                break

            skip_section = False

            section_header_size = extract_data(ByteFormat.INT_32)
            section_header_version = extract_data(ByteFormat.INT_32)
            section_name = extract_data(ByteFormat.WIDE_CHAR, 32).strip()
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
                data_offset = current_position + section_header_size

                if section_name == "Position":
                    tipbox_values = np.fromfile(
                        file_path,
                        map_data_type(section_format, section_bit_size),
                        6,
                        offset=data_offset,
                    )
                    if tipbox_values.size == 6:
                        tipbox = tipbox_values.reshape(2, 3)
                        data_offset += int(tipbox_values.nbytes)

                section_data = np.fromfile(
                    file_path,
                    map_data_type(section_format, section_bit_size),
                    total_items,
                    offset=data_offset,
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
            print(
                f"Section: {section} - {data_sections[section].shape} - {data_sections[section].dtype} - {data_sections[section]}"
            )
    df = pd.DataFrame(data_sections)
    if tipbox is not None:
        df.attrs["tipbox"] = tipbox

    return df
