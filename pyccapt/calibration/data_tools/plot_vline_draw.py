import numpy as np


class VerticalLineManager:
    def __init__(self, variables, ax, fig, x, y):
        self.ax = ax
        self.fig = fig
        self.lines = []
        self.variables = variables
        self.active_line = None
        self.x = x
        self.y = y
        self.shift_is_pressed = False
        self.ctrl_is_pressed = False

        self.is_zooming = False
        self.zoom_factor = 1.1
        if len(x) > 0:
            self.ax.set_xlim(np.min(x), np.max(x))
        self.ax.set_yscale('log')

    def on_press(self, event):
        print('on_press')
        if self.ctrl_is_pressed and event.button == 1:  # if control is pressed and left mouse button is clicked
            closest_line = min(self.lines, key=lambda line: abs(line.get_xdata()[0] - event.xdata))
            self.active_line = closest_line
        elif event.button == 1:  # 1 corresponds to left mouse button
            if not any(abs(line.get_xdata()[0] - event.xdata) < 0.1 for line in self.lines):
                self.active_line = self.ax.axvline(event.xdata, color='b', linestyle='--', linewidth=2)
                self.lines.append(self.active_line)
                self.update_variables(event.xdata)
                self.fig.canvas.draw()
        elif event.button == 3:  # 3 corresponds to right mouse button
            for line in self.lines:
                if abs(line.get_xdata()[0] - event.xdata) < 0.1:  # Note the change here
                    line.remove()
                    self.lines.remove(line)
                    # Find the value in h_line_pos that's closest to line.get_xdata()[0]
                    closest_value = min(self.variables.h_line_pos, key=lambda val: abs(val - line.get_xdata()[0]))

                    self.variables.h_line_pos.remove(closest_value)
                    self.variables.h_line_pos.sort()
                    self.fig.canvas.draw()
                    break

    def update_variables(self, line_pos):
        if 'h_line_pos' in dir(self.variables):
            self.variables.h_line_pos.append(line_pos)
        else:
            self.variables.h_line_pos = [line_pos]
        self.variables.h_line_pos.sort()

    def on_release(self, event):
        self.active_line = None

    def on_motion(self, event):
        if self.active_line:
            self.active_line.set_xdata(event.xdata)
            self.fig.canvas.draw()

    def on_key_press(self, event):
        if event.key == 'shift':
            self.shift_is_pressed = True
        elif event.key == 'r':
            for line in self.lines:
                line.remove()
            self.lines.clear()
            self.variables.h_line_pos.clear()
            self.fig.canvas.draw()
        elif event.key == 'control':
            self.ctrl_is_pressed = True

    def on_key_release(self, event):
        if event.key == 'shift':
            self.shift_is_pressed = False
        elif event.key == 'control':
            self.ctrl_is_pressed = False

    def on_scroll(self, event):
        if self.shift_is_pressed:
            xdata = event.xdata
            left, right = self.ax.get_xlim()
            if event.button == 'up':
                left = xdata - (xdata - left) / self.zoom_factor
                right = xdata + (right - xdata) / self.zoom_factor
            else:  # event.button == 'down'
                left = xdata - (xdata - left) * self.zoom_factor
                right = xdata + (right - xdata) * self.zoom_factor
            self.ax.set_xlim(left, right)
            self.fig.canvas.draw()

    # remove the vertical line
    def remove_all_lines(self):
        if self.lines:
            for line in self.lines:
                line.remove()
            self.lines.clear()
            self.variables.h_line_pos.clear()
            self.fig.canvas.draw()


class HorizontalZoom:
    def __init__(self, ax, fig):
        self.ax = ax
        self.fig = fig
        self.shift_is_pressed = False
        self.zoom_factor = 1.1

    def on_key_press(self, event):
        print('key pressed')
        if event.key == 'shift':
            print('shift pressed')
            self.shift_is_pressed = True

    def on_key_release(self, event):
        if event.key == 'shift':
            self.shift_is_pressed = False

    def on_scroll(self, event):
        if self.shift_is_pressed:
            xdata = event.xdata
            left, right = self.ax.get_xlim()
            if event.button == 'up':
                left = xdata - (xdata - left) / self.zoom_factor
                right = xdata + (right - xdata) / self.zoom_factor
            else:  # event.button == 'down'
                left = xdata - (xdata - left) * self.zoom_factor
                right = xdata + (right - xdata) * self.zoom_factor
            self.ax.set_xlim(left, right)
            self.fig.canvas.draw()
