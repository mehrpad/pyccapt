import itertools
import re
import matplotlib
import numpy as np
import pandas as pd
from faker import Factory

from pyccapt.calibration.data_tools import data_tools


def fix_parentheses(c):
    """
    Fix parentheses in a given string by expanding the elements inside them.

    Args:
        c (str): Input string containing parentheses.

    Returns:
        str: String with expanded elements inside parentheses.
    """
    index = []
    for i in range(len(c)):
        if c[i] == '(':
            index.append(i + 1)
        if c[i] == ')':
            index.append(i)
            index.append(int(c[i + 1]))
    index = list(chunks(index, 3))
    list_parentheses = []
    for i in range(len(index)):
        tmp = c[index[i][0]:index[i][1]]
        tmp = re.findall('[A-Z][^A-Z]*', tmp)
        for j in range(len(tmp)):
            if tmp[j].isalpha():
                tmp[j] = tmp[j] + str(index[i][-1])
            elif not tmp[j].isalpha():
                dd = int(re.findall(r'\d+', tmp[j])[0]) * index[i][-1]
                tmp[j] = ''.join([p for p in tmp[j] if not p.isdigit()]) + str(dd)
        list_parentheses.append("".join(tmp))

    for i in range(len(list_parentheses)):
        gg = list_parentheses[i]
        c = list(c)
        c[index[i][0] - 1 - (2 * i):index[i][1] + 2] = list_parentheses[i]

    return ''.join(c)


def create_formula_latex(aa, num_charge=0):
    """
    Create a LaTeX representation of a chemical formula.

    Args:
        aa (str): The chemical formula.
        num_charge (int): The number of charges associated with the formula.

    Returns:
        str: The LaTeX representation of the chemical formula.

    """
    aa = list(aa)
    for i in range(len(aa)):
        if aa[i] == ')':
            if i + 1 == len(aa):
                aa.insert(i + 1, '1')
            else:
                if not aa[i + 1].isnumeric():
                    aa.insert(i + 1, '1')
    aa = ''.join(aa)
    aa = re.findall('(\d+|[A-Za-z]+)', aa)
    for i in range(int(len(aa) / 3)):
        if aa[i * 3 + 2].isnumeric():
            aa[i * 3 + 2] = int(aa[i * 3 + 2])
    for i in range(len(aa)):
        if aa[i] == '1':
            aa[i] = ' '
    for i in range(int(len(aa) / 3)):
        if i == 0:
            bb = '{}^{%s}%s_{%s}' % (aa[(i * 3) + 1], aa[(i * 3)], aa[(i * 3) + 2])
        else:
            bb += '{}^{%s}%s_{%s}' % (aa[(i * 3) + 1], aa[(i * 3)], aa[(i * 3) + 2])
    if num_charge == 0:
        bb = r'$' + bb + '$'
    else:
        bb = r'$(' + bb + ')^{%s+}' % num_charge + '$'
    return bb


def chunks(lst, n):
    """
    Yield successive n-sized chunks from a list.

    Args:
        lst (list): The input list.
        n (int): The chunk size.

    Yields:
        list: Successive n-sized chunks from the input list.

    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def find_closest_elements(target_elem, num_elements, abundance_threshold=0.0, charge=4, variables=None):
    """
    Find the closest elements to a target element.

    Args:
        target_elem (float): Target element.
        num_elements (int): Number of closest elements to find.
        abundance_threshold (float): Abundance threshold for filtering elements (as a percentage).
        charge (int): Charge value.
        variables (object): Object containing the variables.

    Returns:
        pd.DataFrame: DataFrame containing closest elements and their properties.
    """
    try:
        data_table = '../../../files/isotopeTable.h5'
        dataframe = pd.read_hdf(data_table)
    except Exception as e:
        try:
            data_table = './pyccapt/files/isotopeTable.h5'
            dataframe = pd.read_hdf(data_table)
        except:
            print("Error loading the file", e)

    # Expand elements based on charge
    elements = dataframe['element'].to_numpy()
    isotope_number = dataframe['isotope'].to_numpy()
    weight = dataframe['weight'].to_numpy()
    abundance = dataframe['abundance'].to_numpy()

    elements = np.repeat(elements, charge)
    isotope_number = np.repeat(isotope_number, charge)
    weights = np.repeat(weight, charge)
    abundance = np.repeat(abundance, charge)
    charge_list = np.array([i % charge + 1 for i in range(len(weights))])

    weights = weights / charge_list

    # Filter elements by abundance threshold
    abundance_threshold *= 100
    mask_abundanc = (abundance > abundance_threshold)
    elements = elements[mask_abundanc]
    isotope_number = isotope_number[mask_abundanc]
    weights = weights[mask_abundanc]
    abundance = abundance[mask_abundanc]

    # Find closest elements
    idxs = np.argsort(np.abs(weights - target_elem))[:num_elements]

    selected_elements = elements[idxs]
    selected_isotope_number = isotope_number[idxs]
    selected_weights = weights[idxs]
    selected_abundance = abundance[idxs]
    selected_charge_list = charge_list[idxs]
    # Create LaTeX formatted element symbols
    element_symbols = []
    for i in range(len(idxs)):
        formula = ''
        formula += '{}^'
        formula += '{%s}' % selected_isotope_number[i]
        formula += '%s' % selected_elements[i]

        if selected_charge_list[i] > 1:
            formula = r'$' + formula + '^{%s+}$' % selected_charge_list[i]
        else:
            formula = r'$' + formula + '^{+}$'
        element_symbols.append(formula)


    selected_elements = [[item] for item in selected_elements]
    complex = np.ones(len(idxs), dtype=int)
    complex = [[item] for item in complex]
    selected_isotope_number = [[item] for item in selected_isotope_number]


    selected_isotope_number = [[np.uint32(float(value[0]))] for value in selected_isotope_number]
    complex = [[np.uint32(float(value[0]))] for value in complex]
    selected_charge_list = [np.uint32(value) for value in selected_charge_list]

    # Create DataFrame
    df = pd.DataFrame({
        'ion': element_symbols,
        'mass': selected_weights,
        'element': selected_elements,
        'complex': complex,
        'isotope': selected_isotope_number,
        'charge': selected_charge_list,
        'abundance': selected_abundance,
    })

    # Sort DataFrame
    df = df.sort_values(by=['mass'], ascending=[True])
    df.reset_index(drop=True, inplace=True)
    # Round the abundance column to 4 decimal places
    df['abundance'] = df['abundance'].round(4)
    # Divide all elements in abundance by 100
    df['abundance'] = df['abundance'] / 100
    df['mass'] = df['mass'].round(4)
    # Backup data if variables provided
    if variables is not None:
        variables.range_data_backup = df.copy()

    return df

def load_elements(target_elements, abundance_threshold=0.0, charge=4, variables=None):
    """
    create a dataframe from the given list of ions.

    Args:
        target_elements (str): Target elements.
        abundance_threshold (float): Abundance threshold for filtering elements (as a percentage).
        charge (int): Charge value.
        data_table (str): Path to the data table (HDF5 file).
        variables (object): Object containing the variables.

    Returns:
        pd.DataFrame: DataFrame containing closest elements and their properties.
    """
    try:
        data_table = '../../../files/isotopeTable.h5'
        dataframe = pd.read_hdf(data_table)
    except:
        try:
            data_table = './pyccapt/files/isotopeTable.h5'
            dataframe = pd.read_hdf(data_table)
        except Exception as e:
            print("Error loading the file", e)

    # Expand elements based on charge
    elements = dataframe['element'].to_numpy()
    isotope_number = dataframe['isotope'].to_numpy()
    weight = dataframe['weight'].to_numpy()
    abundance = dataframe['abundance'].to_numpy()


    elements = np.repeat(elements, charge)
    isotope_number = np.repeat(isotope_number, charge)
    weights = np.repeat(weight, charge)
    abundance = np.repeat(abundance, charge)
    charge_list = np.array([i % charge + 1 for i in range(len(weights))])

    weights = weights / charge_list

    # Filter elements by abundance threshold
    abundance_threshold *= 100
    mask_abundanc = (abundance > abundance_threshold)
    elements = elements[mask_abundanc]
    isotope_number = isotope_number[mask_abundanc]
    weights = weights[mask_abundanc]
    abundance = abundance[mask_abundanc]

    target_elements = target_elements.split(',')
    target_elements = [s.replace(' ', '') for s in target_elements]

    index_elements = []
    for i in range(len(target_elements)):
        index_elements.append([index for index, element in enumerate(elements) if element == target_elements[i]])
    idxs = [item for sublist in index_elements for item in sublist]

    selected_elements = elements[idxs]
    selected_isotope_number = isotope_number[idxs]
    selected_weights = weights[idxs]
    selected_abundance = abundance[idxs]
    selected_charge_list = charge_list[idxs]
    # Create LaTeX formatted element symbols
    element_symbols = []
    for i in range(len(idxs)):
        formula = ''
        formula += '{}^'
        formula += '{%s}' % selected_isotope_number[i]
        formula += '%s' % selected_elements[i]

        if selected_charge_list[i] > 1:
            formula = r'$' + formula + '^{%s+}$' % selected_charge_list[i]
        else:
            formula = r'$' + formula + '^{+}$'
        element_symbols.append(formula)

    selected_elements = [[item] for item in selected_elements]
    complex = np.ones(len(idxs), dtype=int)
    complex = [[item] for item in complex]
    selected_isotope_number = [[item] for item in selected_isotope_number]

    complex = [[np.uint32(float(value[0]))] for value in complex]
    selected_isotope_number = [[np.uint32(float(value[0]))] for value in selected_isotope_number]
    selected_charge_list = [np.uint32(value) for value in selected_charge_list]

    # Create DataFrame
    df = pd.DataFrame({
        'ion': element_symbols,
        'mass': selected_weights,
        'element': selected_elements,
        'complex': complex,
        'isotope': selected_isotope_number,
        'charge': selected_charge_list,
        'abundance': selected_abundance,
    })

    # Sort DataFrame
    df = df.sort_values(by=['mass'], ascending=[True])
    df.reset_index(drop=True, inplace=True)

    # Round the abundance column to 4 decimal places
    df['abundance'] = df['abundance'].round(4)
    # Divide all elements in abundance by 100
    df['abundance'] = df['abundance'] / 100
    df['mass'] = df['mass'].round(4)
    # Backup data if variables provided
    if variables is not None:
        variables.range_data_backup = df.copy()

    return df
def molecule_manual(target_element, charge, latex=True, variables=None):
    """
    Generate a list of isotopes for a given target element.

    Args:
        target_element (str): The target element to find isotopes for.
        charge (int): The charge of the target element.
        latex (bool, optional): Whether to generate LaTeX representation of formulas. Defaults to True.
        variables (object, optional): The variables object. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the list of isotopes with their weights and abundances.

    """
    try:
        isotopeTableFile = '../../../files/isotopeTable.h5'
        dataframe = pd.read_hdf(isotopeTableFile, mode='r')
    except:
        try:
            isotopeTableFile = './pyccapt/files/isotopeTable.h5'
            dataframe = pd.read_hdf(isotopeTableFile, mode='r')
        except Exception as e:
            print("Error loading the file", e)
    target_element = fix_parentheses(target_element)

    elements = dataframe['element'].to_numpy()
    isotope_number = dataframe['isotope'].to_numpy()
    abundance = dataframe['abundance'].to_numpy()
    weight = dataframe['weight'].to_numpy()
    # Extract numbers enclosed in curly braces and store them in a list
    isotope_list = [int(match.group(1)) for match in re.finditer(r'{(\d+)}', target_element)]
    # Extract uppercase letters and store them in a list
    element_list = re.findall(r'[A-Z][a-z]*', target_element)

    # Extract numbers that follow the uppercase letters and store them in a list
    complexity_list = [int(match[1]) for match in re.findall(r'([A-Z][a-z]*)(\d+)', target_element)]

    total_weight = 0
    abundance_c = 1
    for i, isotop in enumerate(isotope_list):
        index = np.where(isotope_number == isotop)
        total_weight += weight[index] * complexity_list[i]
        abundance_c *= (abundance[index] / 100) ** complexity_list[i]

    total_weight = total_weight / charge
    if latex:
        formula = ''
        for i in range(len(isotope_list)):
            isotope = isotope_list[i]
            element = element_list[i]
            comp = complexity_list[i]

            formula += '{}^'
            formula += '{%s}' % isotope
            formula += '%s' % element
            if comp != 1:
                formula += '_{%s}' % comp
        if charge > 1:
            formula = r'$(' + formula + ')^{%s+}$' % charge
        else:
            formula = r'$' + formula + ')^{+}$'
    else:
        formula = target_element

    # Convert float_list to a list of uint32 values
    complexity_list = [np.uint32(value) for value in complexity_list]
    isotope_list = [np.uint32(value) for value in isotope_list]
    charge = np.uint32(charge)

    element_list = [element_list]
    complexity_list = [complexity_list]
    isotope_list = [isotope_list]
    charge = [charge]

    df = pd.DataFrame({'ion': formula, 'mass': total_weight, 'element': element_list,
                       'complex': complexity_list, 'isotope': isotope_list, 'charge': charge,
                       'abundance': abundance_c, })


    # Round the abundance column to 4 decimal places
    df['abundance'] = df['abundance'].round(4)
    df['mass'] = df['mass'].round(4)

    if variables is not None:
        variables.range_data_backup = df.copy()
    return df


def transform_combination_and_isotopes(combination, isotopes):
    """
    Transform the combination and isotopes lists to remove duplicates.

    Args:
        combination (list): The list of elements.
        isotopes (list): The list of isotopes.

    Returns:
        list: The new combination list.
    """
    new_combination = []
    new_isotopes = []
    complexity = []

    for element, isotope in zip(combination, isotopes):
        if element not in new_combination:
            new_combination.append(element)
            new_isotopes.append(isotope)
            complexity.append(1)
        else:
            index = new_combination.index(element)
            if isotope != new_isotopes[index]:
                new_combination.append(element)
                new_isotopes.append(isotope)
                complexity.append(1)
            else:
                complexity[index] += 1

    return new_combination, new_isotopes, complexity


def molecule_create(element_list, max_complexity, charge, abundance_threshold, variables=None, latex=True):
    """
    Generate a list of isotopes for a given target element.

    Args:
        element_list (str): The target element to find isotopes for.
        max_complexity (int): The maximum complexity of the molecule.
        charge (int): The charge of the target element.
        abundance_threshold (float): The abundance threshold for filtering isotopes.
        variables (object, optional): The variables object. Defaults to None.
        latex (bool, optional): Whether to generate LaTeX representation of formulas. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the list of isotopes with their weights and abundances.
    """
    try:
        isotopeTableFile = '../../../files/isotopeTable.h5'
        dataframe = data_tools.read_range(isotopeTableFile)
    except:
        try:
            isotopeTableFile = './pyccapt/files/isotopeTable.h5'
            dataframe = data_tools.read_range(isotopeTableFile)
        except Exception as e:
            print("Error loading the file", e)
    elements = dataframe['element'].to_numpy()
    isotope_number = dataframe['isotope'].to_numpy()
    abundance = dataframe['abundance'].to_numpy()
    weight = dataframe['weight'].to_numpy()
    element_list = element_list.split(',')
    element_list = [s.replace(' ', '') for s in element_list]
    indices_elements = np.where(np.isin(elements, [element_list]))
    selected_elements = elements[indices_elements[0]]
    selected_isotope_number = isotope_number[indices_elements[0]]
    selected_weights = weight[indices_elements[0]]
    selected_abundance = abundance[indices_elements[0]]

    # Create a list of elements with their respective isotope numbers, weights, and abundances
    element_data = [{'element': elem, 'isotope': iso, 'weight': w, 'abundance': ab}
                    for elem, iso, w, ab in zip(selected_elements, selected_isotope_number, selected_weights,
                                                selected_abundance)]
    # Initialize lists to store results
    combinations = []
    combinations_list = []
    combination_weights = []
    combination_isotopes = []
    combination_abundances = []
    combination_charge = []
    combination_complexity = []
    combination_formula = []
    # Generate all possible combinations with complexity 3
    for comploex in range(1, max_complexity + 1):
        for combo in itertools.product(element_data, repeat=comploex):
            combo_elements = [elem['element'] for elem in combo]
            combo_weights = [elem['weight'] for elem in combo]
            combo_abundances = [elem['abundance'] for elem in combo]

            combo_weight = sum(combo_weights)
            combo_abundance = 1.0

            for ab in combo_abundances:
                combo_abundance *= ab / 100
            for i in range(charge):
                combinations.append(combo_elements)
                combinations_list.append(combo_elements)
                combination_weights.append(combo_weight / (i + 1))
                combination_isotopes.append([elem['isotope'] for elem in combo])
                combination_abundances.append(combo_abundance)
                combination_charge.append(i + 1)

    for i in range(len(combinations)):
        new_combination, new_isotopes, complexity = transform_combination_and_isotopes(combinations[i],
                                                                                       combination_isotopes[i])
        combinations[i] = new_combination
        combination_isotopes[i] = new_isotopes
        combination_complexity.append(complexity)
        charge = combination_charge[i]
        if latex:
            formula = ''
            for i in range(len(new_isotopes)):
                isotope = new_isotopes[i]
                element = new_combination[i]
                comp = complexity[i]

                formula += '{}^'
                formula += '{%s}' % isotope
                formula += '%s' % element
                if comp != 1:
                    formula += '_{%s}' % comp
            if charge > 1:
                formula = r'$' + formula + '^{%s+}$' % charge
            else:
                formula = r'$' + formula + '^{+}$'

        else:
            element_counts = {}
            chemical_formula = ''

            for element in element_list:
                if element in element_counts:
                    element_counts[element] += 1
                else:
                    element_counts[element] = 1

            for element, count in element_counts.items():
                chemical_formula += element
                if count > 1:
                    chemical_formula += str(count)
            formula = chemical_formula
        combination_formula.append(formula)

    combination_complexity = [[np.uint32(x) for x in sub_list] for sub_list in combination_complexity]
    combination_isotopes = [[np.uint32(x) for x in sub_list] for sub_list in combination_isotopes]
    combination_charge = [np.uint32(value) for value in combination_charge]

    # Create DataFrame
    df = pd.DataFrame({
        'ion': combination_formula,
        'mass': combination_weights,
        'element': combinations,
        'complex': combination_complexity,
        'isotope': combination_isotopes,
        'charge': combination_charge,
        'abundance': combination_abundances,
    })

    df = df[df['abundance'] > abundance_threshold]

    # Reset the index
    df = df.reset_index(drop=True)

    # Sort DataFrame
    df = df.sort_values(by=['mass'], ascending=[True])
    df.reset_index(drop=True, inplace=True)

    # Round the abundance column to 4 decimal places
    df['abundance'] = df['abundance'].round(4)
    df['mass'] = df['mass'].round(4)
    # Backup data if variables provided
    if variables is not None:
        variables.range_data_backup = df.copy()

    return df


def ranging_dataset_create(variables, row_index, mass_ion):
    """
    This function is used to create the ranging dataset

        Arg:
            variables (class): The class of the variables
            row_index (int): The index of the selected row
            mass_ion (float): The mass of the element

        Returns:
            None
    """
    if len(variables.range_data_backup) == 0:
        print('The dataframe of elements is empty')
        print('First press the button to find the closest elements')
    else:
        if row_index >= 0:
            selected_row = variables.range_data_backup.iloc[row_index].tolist()
            selected_row = selected_row[:-1]
        else:
            selected_row = ['un', mass_ion, ['unranged'], [0], [0], 0]
        fake = Factory.create()
        try:
            data_table = '../../../files/color_scheme.h5'
            dataframe = data_tools.read_range(data_table)
        except:
            try:
                data_table = './pyccapt/files/color_scheme.h5'
                dataframe = data_tools.read_range(data_table)
            except Exception as e:
                print("Error loading the file", e)
        element_selec = selected_row[2]
        if len(element_selec) == 1:
            element_selec = element_selec[0]
            try:
                color_rgb = dataframe[dataframe['ion'].str.contains(element_selec, na=False)].to_numpy().tolist()
                color = matplotlib.colors.to_hex([color_rgb[0][1], color_rgb[0][2], color_rgb[0][3]])
                print(f"Color: {color}")
            except Exception as e:
                print(f"An error occurred: {e}")
                print('The element is not color list')
                color = fake.hex_color()
        else:
            color = fake.hex_color()
        if not variables.h_line_pos:
            print('The h_line_pos is empty')
            print('First specify the left and right boundary for the selected peak')
        else:
            # Find the closest h_line that is smaller than mass
            smaller_h_line = max(filter(lambda x: x < mass_ion, variables.h_line_pos))
            # Find the closest h_line that is bigger than mass
            bigger_h_line = min(filter(lambda x: x > mass_ion, variables.h_line_pos))

            def generate_name(elements, counts):
                return ".".join(f"{el}{ct}" for el, ct in zip(elements, counts))

            name = generate_name(selected_row[2], selected_row[3])
            selected_row.insert(0, name)
            selected_row.insert(3, mass_ion)
            selected_row.insert(4, smaller_h_line)
            selected_row.insert(5, bigger_h_line)
            selected_row.insert(6, color)

            # Add the row to the DataFrame using the .loc method
            selected_row[9] = np.uint32(selected_row[9])
            print(f"Selected row: {selected_row}")
            variables.range_data.loc[len(variables.range_data)] = selected_row

def display_color(color):
    """
    This function is used to display the color in the table

        Arg:
            color (str): The color in hex format

        Returns:
            str: The color in hex format
    """
    return f'background-color: {color}; width: 50px; height: 20px;'