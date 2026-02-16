"""Contains the configuration application.

The config app enables adjusting parameters of the whippersnappy program
during runtime using a simple GUI window.

Dependencies:
    PyQt6

@Author    : Ahmed Faisal Abdelrahman
@Created   : 20.03.2022

"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class ConfigWindow(QWidget):
    """Qt configuration window for interactive parameter tuning.

    The configuration window exposes sliders and text boxes to adjust the
    f-threshold and f-max parameters used by the renderer. The widget is
    intended to run alongside the OpenGL window and to push updated values
    to the renderer via polling from the main loop.

    Parameters
    ----------
    parent : QWidget, optional
        Parent Qt widget. Defaults to ``None``.
    screen_dims : tuple or None, optional
        (width, height) of the available screen; used to position the
        window in the top-right corner when provided.
    initial_fthresh_value : float, optional
        Initial threshold value (default 2.0).
    initial_fmax_value : float, optional
        Initial fmax value (default 4.0).
    """

    def __init__(
        self,
        parent=None,
        screen_dims=None,
        initial_fthresh_value=2.0,
        initial_fmax_value=4.0,
    ):
        super().__init__(parent)

        self.current_fthresh_value = initial_fthresh_value
        self.current_fmax_value = initial_fmax_value
        self.screen_dims = screen_dims
        self.window_size = (400, 200)

        layout = QVBoxLayout()

        # fthresh slider:
        self.fthresh_slider_tick_limits = (0.0, 1000.0)
        self.fthresh_slider_value_limits = (0.0, 10.0)
        self.fthresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.fthresh_slider.setMinimum(int(self.fthresh_slider_tick_limits[0]))
        self.fthresh_slider.setMaximum(int(self.fthresh_slider_tick_limits[1]))
        self.fthresh_slider.setValue(
            int(
                self.convert_value_to_range(
                    self.current_fthresh_value,
                    self.fthresh_slider_value_limits,
                    self.fthresh_slider_tick_limits,
                )
            )
        )
        self.fthresh_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.fthresh_slider.setTickInterval(
            int(
                (
                    self.fthresh_slider_tick_limits[1]
                    - self.fthresh_slider_tick_limits[0]
                )
                / 20.0
            )
        )
        self.fthresh_slider.valueChanged.connect(self.fthresh_slider_value_cb)

        # fthresh value input box:
        self.fthresh_value_box = QLineEdit("Threshold")
        self.fthresh_value_box.setText(str(self.current_fthresh_value))
        self.fthresh_value_box.textChanged.connect(self.fthresh_value_cb)

        # fmax slider:
        self.fmax_slider_tick_limits = (0.0, 1000.0)
        self.fmax_slider_value_limits = (0.0, 10.0)
        self.fmax_slider = QSlider(Qt.Orientation.Horizontal)
        self.fmax_slider.setMinimum(int(self.fmax_slider_tick_limits[0]))
        self.fmax_slider.setMaximum(int(self.fmax_slider_tick_limits[1]))
        self.fmax_slider.setValue(
            int(
                self.convert_value_to_range(
                    self.current_fmax_value,
                    self.fmax_slider_value_limits,
                    self.fmax_slider_tick_limits,
                )
            )
        )
        self.fmax_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.fmax_slider.setTickInterval(
            int(
                (self.fmax_slider_tick_limits[1] - self.fmax_slider_tick_limits[0])
                / 20.0
            )
        )
        self.fmax_slider.valueChanged.connect(self.fmax_slider_value_cb)

        # fmax value input box:
        self.fmax_value_box = QLineEdit("Threshold")
        self.fmax_value_box.setText(str(self.current_fmax_value))
        self.fmax_value_box.textChanged.connect(self.fmax_value_cb)

        # Group boxes and widget layout:
        fthresh_box = QGroupBox("Thresh")
        fthresh_box_layout = QVBoxLayout()
        fthresh_box_layout.addWidget(self.fthresh_slider)
        fthresh_box_layout.addWidget(self.fthresh_value_box)
        fthresh_box.setLayout(fthresh_box_layout)
        fthresh_box.setStyleSheet("QGroupBox  {color: blue;}")

        fmax_box = QGroupBox("Max")
        fmax_box_layout = QVBoxLayout()
        fmax_box_layout.addWidget(self.fmax_slider)
        fmax_box_layout.addWidget(self.fmax_value_box)
        fmax_box.setLayout(fmax_box_layout)
        fmax_box.setStyleSheet("QGroupBox  {color: green;}")

        thresholds_box = QGroupBox("Thresholds")
        thresholds_box_layout = QHBoxLayout()
        thresholds_box_layout.addWidget(fthresh_box)
        thresholds_box_layout.addWidget(fmax_box)
        thresholds_box.setLayout(thresholds_box_layout)

        layout.addWidget(thresholds_box)

        # Main window configurations:
        self.setWindowTitle("WhipperSnapPy Configuration")
        self.setLayout(layout)
        if self.screen_dims is not None:
            self.setGeometry(
                screen_dims[0] - int(screen_dims[0] / 5),
                int(screen_dims[1] / 8),
                self.window_size[0],
                self.window_size[1],
            )
        else:
            self.setGeometry(0, 0, self.window_size[0], self.window_size[1])

    def fthresh_slider_value_cb(self):
        """Handle changes from the f-threshold slider.

        This slot is connected to the slider's valueChanged signal. It maps the
        slider tick value into the configured value range and updates the
        text input box accordingly.
        """
        self.current_fthresh_value = self.convert_value_to_range(
            self.fthresh_slider.value(),
            self.fthresh_slider_tick_limits,
            self.fthresh_slider_value_limits,
        )
        self.fthresh_value_box.setText(str(self.current_fthresh_value))

    def fthresh_value_cb(self, new_value):
        """Handle text input changes for f-threshold.

        Parameters
        ----------
        new_value : float or str
            The new value input by the user. May be a float or numeric string.
        """
        # Do not react to invalid values:
        try:
            new_value = float(new_value)
        except ValueError:
            return

        self.current_fthresh_value = float(new_value)

        slider_fthresh_value = self.convert_value_to_range(
            self.current_fthresh_value,
            self.fthresh_slider_value_limits,
            self.fthresh_slider_tick_limits,
        )
        self.fthresh_slider.setValue(int(slider_fthresh_value))

    def fmax_slider_value_cb(self):
        """Handle changes from the f-max slider and update the text box."""
        self.current_fmax_value = self.convert_value_to_range(
            self.fmax_slider.value(),
            self.fmax_slider_tick_limits,
            self.fmax_slider_value_limits,
        )
        self.fmax_value_box.setText(str(self.current_fmax_value))

    def fmax_value_cb(self, new_value):
        """Handle text input changes for f-max.

        Parameters
        ----------
        new_value : float or str
            New value provided by the user.
        """
        # Do not react to invalid values:
        try:
            new_value = float(new_value)
        except ValueError:
            return

        self.current_fmax_value = float(new_value)

        slider_fmax_value = self.convert_value_to_range(
            self.current_fmax_value,
            self.fmax_slider_value_limits,
            self.fmax_slider_tick_limits,
        )
        self.fmax_slider.setValue(int(slider_fmax_value))

    def convert_value_to_range(self, value, old_limits, new_limits):
        """Map ``value`` from ``old_limits`` to ``new_limits``.

        Parameters
        ----------
        value : float
            Value to be converted.
        old_limits : tuple
            (min, max) source range.
        new_limits : tuple
            (min, max) target range.

        Returns
        -------
        float
            Value mapped into ``new_limits``.
        """
        old_range = old_limits[1] - old_limits[0]
        new_range = new_limits[1] - new_limits[0]
        new_value = (((value - old_limits[0]) * new_range) / old_range) + new_limits[0]

        return new_value

    def get_fthresh_value(self):
        """Return the currently selected f-threshold value.

        Returns
        -------
        float
            Current f-threshold value.
        """
        return self.current_fthresh_value

    def get_fmax_value(self):
        """Return the currently selected f-max value.

        Returns
        -------
        float
            Current f-max value.
        """
        return self.current_fmax_value

    def keyPressEvent(self, event):
        """Handle key press events for the window.

        The handler closes the window when the ESC key is pressed.

        Parameters
        ----------
        event : QKeyEvent
            Qt key event delivered by the framework.
        """
        if event.key() == Qt.Key.Escape:
            self.close()
