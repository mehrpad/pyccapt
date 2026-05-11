from pyccapt.control.devices.pfeiffer_gauges import TPG362


if __name__ == '__main__':
    tpg = TPG362(port='COM5')

    value, _ = tpg.pressure_gauge(2)
    unit = tpg.pressure_unit()
    print('pressure is {} {}'.format(value, unit))
