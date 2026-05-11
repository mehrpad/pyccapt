import sys

import pandas as pd
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QFont, QGuiApplication
from PyQt6.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


class MyWindow(QMainWindow):

    def __init__(self, data, variables):
        super().__init__()
        self.data = data
        self.variables = variables
        # Extract the column names and save them in a list
        self.column_names = data.columns.tolist()
        self.setWindowTitle("Ions Table")
        self.setGeometry(100, 100, 800, 400)

        self.initUI()

    def initUI(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.tableWidget = QTableWidget(self)
        self.tableWidget.setRowCount(len(self.data['ion']))
        self.tableWidget.setColumnCount(len(self.data.columns))  # Plus one for the headers

        # Add the first row for column headers
        for col, header in enumerate(self.data.columns):
            header_item = QTableWidgetItem(header)
            header_font = QFont()
            header_font.setBold(True)
            header_item.setFont(header_font)
            self.tableWidget.setItem(0, col, header_item)
        # ``QWebEngineView`` lives in the optional ``PyQt6.QtWebEngineWidgets``
        # module (``PyQt6-WebEngine`` package). We only need it for cells
        # that contain a ``$…$`` LaTeX formula, so we do a lazy import inside
        # the loop and fall back to plain text when the package is missing —
        # the rest of the dialog still renders.
        try:
            from PyQt6.QtWebEngineWidgets import QWebEngineView   # noqa: F401
            webengine_available = True
        except ImportError:
            webengine_available = False

        for row, (_, row_data) in enumerate(self.data.iterrows()):
            for col, value in enumerate(row_data):
                is_latex = (
                    isinstance(value, str)
                    and value.startswith('$')
                    and value.endswith('$')
                )
                if is_latex and webengine_available:
                    # Remove the surrounding ``$``s and embed the formula in a
                    # one-liner MathJax page, then render it as a cell widget.
                    formula = value[1:-1]
                    webview = QWebEngineView(self)
                    webview.setHtml(
                        '<html><head><script type="text/javascript" async '
                        'src="https://cdn.mathjax.org/mathjax/latest/'
                        'MathJax.js?config=TeX-MML-AM_CHTML"></script></head>'
                        f'<body>\\({formula}\\)</body></html>'
                    )
                    formula_item = QTableWidgetItem()
                    formula_item.setFlags(
                        formula_item.flags() & ~Qt.ItemFlag.ItemIsEditable
                    )
                    self.tableWidget.setCellWidget(row + 1, col, webview)
                else:
                    # Either the value isn't a LaTeX formula, or PyQt6-WebEngine
                    # isn't installed — fall back to the raw text so the rest
                    # of the table is still usable.
                    item = QTableWidgetItem(str(value))
                    self.tableWidget.setItem(row + 1, col, item)
        layout.addWidget(self.tableWidget)

        # Set row height for all rows
        for row in range(self.tableWidget.rowCount()):
            self.tableWidget.setRowHeight(row, 40)
        # Set column width for the 'Formula' column (column 0)
        self.tableWidget.setColumnWidth(0, 200)
        # self.tableWidget.resizeColumnsToContents()
        # self.tableWidget.resizeRowsToContents()

        # Connect the selectionChanged signal to a custom slot
        self.tableWidget.itemSelectionChanged.connect(self.onSelectionChanged)

        # Connect column headers to sorting function
        self.tableWidget.horizontalHeader().sectionClicked.connect(self.sortByColumn)

        # Disable selection for column headers
        # self.tableWidget.horizontalHeader().setSectionsClickable(False)

    def onSelectionChanged(self):
        # Get the selected row and print its data
        selected_items = self.tableWidget.selectedItems()
        header = False
        # Check if any selected items are in the data region (not in the header row or index column)
        for item in selected_items:
            if item.row() == 0:
                header = True

        if not header:
            new_selected_row = selected_items[0].row()
            selected_row = selected_items[0].row()

            selected_data = self.data.iloc[selected_row - 1:selected_row]

            print(selected_data)

            self.selected_row = new_selected_row

            # List to match for 'element' and 'complex'
            target_element_list = selected_data['element'].to_list()
            target_element_list = [item for sublist in target_element_list for item in sublist]
            target_complex_list = selected_data['complex'].to_list()
            target_complex_list = [item for sublist in target_complex_list for item in sublist]
            # Find rows with matching 'element' and 'complex' lists
            matching_rows = self.data[
                (self.data['element'].apply(lambda x: all(item in x for item in target_element_list))) &
                (self.data['complex'].apply(lambda x: all(item in x for item in target_complex_list)))
                ]
            # Create a new DataFrame with matching rows
            new_df = pd.DataFrame(matching_rows)
            # Reset the index of the new DataFrame if needed
            new_df.reset_index(drop=True, inplace=True)
            self.variables.ions_list_data = new_df
            # self.variables.AptHistPlotter.plot_founded_range_loc(new_df)

            # Disconnect the selectionChanged signal to a custom slot
            self.tableWidget.itemSelectionChanged.disconnect(self.onSelectionChanged)

            for col in range(self.tableWidget.columnCount()):
                item = self.tableWidget.item(self.selected_row, col)
                if item is not None:
                    item.setSelected(True)  # Highlight the item

            # Connect the selectionChanged signal to a custom slot
            self.tableWidget.itemSelectionChanged.connect(self.onSelectionChanged)

    def sortByColumn(self, column):
        # Disable selection for column headers
        self.tableWidget.horizontalHeader().setSectionsClickable(False)
        # Disconnect the selectionChanged signal to a custom slot
        self.tableWidget.itemSelectionChanged.disconnect(self.onSelectionChanged)

        # Extract data from the table
        data = []
        for row in range(1, self.tableWidget.rowCount()):
            row_data = []
            for col in range(self.tableWidget.columnCount()):
                item = self.tableWidget.item(row, col)
                if item is not None:
                    row_data.append(item.text())
                else:
                    row_data.append(None)
            data.append(row_data)

        # Sort the data by the selected column
        data.sort(key=lambda row: row[column])

        # Update the table with the sorted data
        for row in range(1, self.tableWidget.rowCount()):
            for col in range(self.tableWidget.columnCount()):
                item = self.tableWidget.item(row, col)
                if item is not None:
                    item.setText(str(data[row - 1][col]))

        # Enable selection for column headers after sorting
        self.tableWidget.horizontalHeader().setSectionsClickable(True)

        # Deselect all rows and columns after sorting
        self.deselectAllRowsAndColumns()
        # Connect the selectionChanged signal to a custom slot
        self.tableWidget.itemSelectionChanged.connect(self.onSelectionChanged)

    def deselectAllRowsAndColumns(self):
        for row in range(self.tableWidget.rowCount()):
            for col in range(self.tableWidget.columnCount()):
                item = self.tableWidget.item(row, col)
                if item is not None:
                    item.setSelected(False)


if __name__ == '__main__':

    def open_gui(df, variables):
        app = 0  # This is the solution to prevent kernel crash of Jupyter lab
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)

        window = MyWindow(df, variables)
        window.show()
        # Ensure that the app is deleted when we close it
        app.aboutToQuit.connect(app.deleteLater)
        try:
            app.exec()
        except SystemExit:
            pass
