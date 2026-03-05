import re

import ipywidgets as widgets
import pandas as pd

from pyccapt.calibration.data_tools import data_tools

# Stores values of the currently selected element in the dropdown.
elementDict = {}

# Stores values of the currently selected charge in the dropdown.
chargeDict = {}

# Stores values of the currently selected element with charge in the dropdown.
elementWithChargeDict = {}


def dropdownWidget(elementsList, dropdownLabel):
    """
    Create a dropdown widget for selecting elements.

    Args:
        elementsList (list): List of elements with their corresponding mass/weight.
                             Format: [('H', 1.01), ('He', 3.02)]
        dropdownLabel (str): Label for the dropdown widget.

    Returns:
        dropdown (object): Dropdown widget object.
    """
    dropdown = widgets.Dropdown(options=elementsList, description=dropdownLabel, disabled=False)

    if dropdownLabel == "Charge":
        chargeDict['charge'] = elementsList[0][1]
    elif dropdownLabel == "Elements":
        elementDict['element'] = elementsList[0][1]

    return dropdown


def buttonWidget(buttonText):
    """
    Create a button widget.

    Args:
        buttonText (str): Text to be displayed on the button.

    Returns:
        button (object): Button widget object.
    """
    button = widgets.Button(
        description=buttonText,
        disabled=False,
        button_style='',
        tooltip=buttonText,
        icon='check'
    )
    return button


def onClickAdd(b, variables):
    """
    Callback function for the ADD button click event.
    Adds the selected element in the dropdown to a list.

    Args:
        variables (object): Object of the Variables class.
    Returns:
        None
    """
    if 'element' in elementWithChargeDict:
        elementMass = elementWithChargeDict['element']
        if elementMass not in variables.list_material:
            variables.list_material.append(elementMass)

            selectedElement = elementDict['element']
            charge = chargeDict['charge']
            variables.charge.append(charge)
            element = re.sub("\d+", "", selectedElement)
            element = re.sub("[\(\[].*?[\)\]]", "", element)
            variables.element.append(element)
            isotope = int(re.findall("\d+", selectedElement)[0])
            variables.isotope.append(isotope)

            print("Updated List: ", variables.list_material)
            print("Updated element List: ", variables.element)
            print("Updated isotope List: ", variables.isotope)
            print("Updated charge List: ", variables.charge)
    else:
        print("Please select the charge before adding", end='\r')


def onClickDelete(b, variables):
    """
    Callback function for the DELETE button click event.
    Deletes the selected element in the dropdown from the list.
    Args:
        variables (object): Variables object.
    Returns:
        None
    """
    if 'element' in elementWithChargeDict:
        elementMass = elementWithChargeDict['element']
        if elementMass in variables.list_material:
            variables.list_material.remove(elementMass)
            variables.element.pop()
            variables.isotope.pop()
            variables.charge.pop()
            print("Updated List: ", variables.list_material)
            print("Updated element List: ", variables.element)
            print("Updated isotope List: ", variables.isotope)
            print("Updated charge List: ", variables.charge)
        else:
            print("Nothing Deleted. Choose carefully (Enter the right combination of element and charge)", end='\r')
    else:
        print("Please select the element with the right combination of charge to efficiently delete", end='\r')


def onClickReset(b, variables):
    """
    Callback function for the RESET button click event.
    Clears the list and deletes all the elements from it.
    Args:
        variables (object): Variables object.
    Returns:
        None
    """
    variables.list_material.clear()
    variables.element.clear()
    variables.isotope.clear()
    variables.charge.clear()
    print("Updated List: ", variables.list_material)
    print("Updated element List: ", variables.element)
    print("Updated isotope List: ", variables.isotope)
    print("Updated charge List: ", variables.charge)


def on_change(change):
    """
    Callback function for observing changes in the dropdown widget.
    Updates the selected element and its corresponding weight/mass based on the dropdown selection.
    """
    if change['type'] == 'change' and change['name'] == 'value':
        print("Mass of selected element: %s" % change['new'], ''.ljust(40), end='\r')
        elementWithChargeDict.clear()
        print("Now please select the appropriate charge", ''.ljust(40), end='\r')
        elementDict['element'] = change['new']
        compute_element_isotope_values_according_to_selected_charge()


def on_change_ions_selection(change):
    """
    Callback function for observing changes in the dropdown widget for ions selection.
    Updates the selected element and its corresponding weight/mass based on the dropdown selection.
    """
    if change['type'] == 'change' and change['name'] == 'value':
        print("Mass of selected element: %s" % change['new'], ''.ljust(40), end='\r')
        elementWithChargeDict.clear()
        print("Now please select the appropriate charge", ''.ljust(40), end='\r')
        elementDict['element'] = change['new']
        compute_element_isotope_values_according_to_selected_charge(mode='ions_selection')


def on_change_charge(change):
    """
    Callback function for observing changes in the dropdown widget for charge selection.
    Updates the selected charge value.
    """
    if change['type'] == 'change' and change['name'] == 'value':
        print("Selected charge: %s" % change['new'], ''.ljust(40), end='\r')
        updatedCharge = change['new']
        chargeDict['charge'] = updatedCharge
        compute_element_isotope_values_according_to_selected_charge()


def on_change_charge_ions_selection(change):
    """
    Callback function for observing changes in the dropdown widget for ions selection and charge.
    Updates the selected charge value.
    """
    if change['type'] == 'change' and change['name'] == 'value':
        print("Selected charge: %s" % change['new'], ''.ljust(40), end='\r')
        updatedCharge = change['new']
        chargeDict['charge'] = updatedCharge
        compute_element_isotope_values_according_to_selected_charge(mode='ions_selection')


def compute_element_isotope_values_according_to_selected_charge(mode='calibration'):
    """
    Compute the element and isotope values based on the selected charge.

    Args:
        mode (str): Computation mode. Defaults to 'calibration'.

    """
    selectedElement = elementDict['element']
    charge = chargeDict['charge']

    if mode == 'calibration':
        elem = re.findall('\[(.*?)\]', selectedElement)
        elementWithCharge = round(float(elem[0]) / int(charge), 2)
        elementWithChargeDict['element'] = elementWithCharge
    elif mode == 'ions_selection':
        elementWithCharge = selectedElement + '(' + str(charge) + '+)'
        elementWithChargeDict['element'] = elementWithCharge


def dataset_instrument_specification_selection():
    """
    Create and return the dataset TDC selection widgets.

    Returns:
        tdc (object): Dropdown widget for selecting data mode.
        pulse_mode (object): Dropdown widget for selecting pulse mode.
        flightPathLength (object): FloatText widget for flight path length.
        t0 (object): FloatText widget for t0.
        max_mc (object): FloatText widget for maximum mc.
        det_diam (object): FloatText widget for detector diameter.
    """
    flightPathLength = widgets.FloatText(
        value='110',
        placeholder='Flight path length',
        description='Flight path length:',
        disabled=False
    )

    det_diam = widgets.FloatText(
        value='80',
        placeholder='Detector diameter',
        description='Detector diameter:',
        disabled=False
    )

    t0 = widgets.FloatText(
        value='38',
        placeholder='T_0 of the instrument',
        description='t0:',
        disabled=False
    )

    max_mc = widgets.FloatText(
        value='400',
        placeholder='Maximum possible mc',
        description='Max mc:',
        disabled=False
    )

    tdc = widgets.Dropdown(
        options=['pyccapt', 'leap_epos', 'leap_pos', 'leap_apt', 'ato_v6'],
        value='pyccapt',
        description='Data mode:',
    )

    pulse_mode = widgets.Dropdown(
        options=['voltage', 'laser'],
        value='voltage',
        description='Pulse mode:',
    )

    return tdc, pulse_mode, flightPathLength, t0, max_mc, det_diam


def density_field_selection():
    """
    Create and return the element dropdown widget for density field selection.

    Returns:
        element (object): Dropdown widget for selecting an element.
    """
    try:
        TableFile = '../../../files/field_density_table.h5'
        dataframe = pd.read_hdf(TableFile, mode='r')
    except Exception as e:
        try:
            TableFile = './pyccapt/files/field_density_table.h5'
            dataframe = pd.read_hdf(TableFile, mode='r')
        except Exception as e:
            print("Error: ", e)
            
    elementsAtomicNumber = dataframe['atomic_number']
    elementsList = dataframe['element']
    elementDensityList = dataframe['atom_density']
    elementFieldList = dataframe['field_evaporation']

    elementsAtomicNumber.to_numpy()
    elements = list(zip(elementsAtomicNumber, elementsList, elementDensityList, elementFieldList))
    dropdownList = []
    for index, element in enumerate(elements):
        tupleElement = (
            "{} - {} - Density({}) - FieldEva({})".format(element[0], element[1], element[2], element[3]),
        )
        dropdownList.append(tupleElement)

    element = widgets.Dropdown(
        options=elements,
        description='Element'
    )
    return element
