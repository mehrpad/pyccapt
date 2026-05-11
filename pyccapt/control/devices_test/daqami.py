from mcculw import ul
from mcculw.enums import TempScale, TInOptions


def run_example():
    """
    Runs the temperature reading example using the MCC Universal Library.
    """
    device_to_show = "USB-TC"
    board_num = 0

    try:
        board_name = ul.get_board_name(board_num)

    except Exception as e:
        if ul.ErrorCode(1):
            print("\nNo board found at Board 0.")
            print(e)
            return

    else:
        if device_to_show in board_name:
            print("{0} found as Board number {1}.\n".format(board_name, board_num))
            ul.flash_led(board_num)
        else:
            print("\nNo {0} series found as Board 0. Please run InstaCal.".format(device_to_show))
            return

    try:
        channel_list = ['MC_NEG', 'MC_Det', 'Mc-Top', 'MC-Gate', 'BC-Top', 'BC-Pump']
        for i, channel_name in enumerate(channel_list):
            options = TInOptions.NOFILTER
            value_temperature = ul.t_in(board_num, i, TempScale.CELSIUS, options)
            print("Channel {:d} - {:s}: {:.3f} Degrees.".format(i, channel_name, value_temperature))

    except Exception as e:
        print('\n', e)


if __name__ == '__main__':
    run_example()
