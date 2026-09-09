#pip install PyQt5 opencv-python numpy
import cv2
import numpy as np
import os
import glob
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QLabel, QSlider, QPushButton, QLineEdit, QCheckBox,
    QFileDialog, QScrollArea, QSplitter, QGroupBox, QSpinBox, QDoubleSpinBox,
    QMessageBox, QComboBox, QShortcut
)
from PyQt5.QtGui import QPixmap, QImage, QKeySequence
from PyQt5.QtCore import Qt, QTimer

# ======================================================================
# -------------------- 1. OpenCV image-processing backend --------------------
# ======================================================================

def adjust_alpha_beta(image, alpha, beta):
    """
    Adjust brightness/contrast.
    :param image: OpenCV image (numpy array).
    :param alpha: contrast (1.0 = unchanged).
    :param beta: brightness (0 = unchanged).
    :return: adjusted image.
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def adjust_gamma(image, gamma):
    """
    Gamma-correct an image (exposure adjustment) using a lookup table.
    :param image: OpenCV image (numpy array).
    :param gamma: gamma value (1.0 = unchanged, <1.0 brighter, >1.0 darker).
    :return: adjusted image.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def find_coin_center_and_size(image, debug_output_path=None):
    """
    Locate the coin's center and size via adaptive thresholding + contour detection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,  # neighborhood size
        2    # constant C
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Warning: No valid contour found. Using image center for cropping.")
        return (image.shape[1] // 2, image.shape[0] // 2), min(image.shape[0], image.shape[1])

    max_contour = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(max_contour)
    (center_x, center_y), (width, height), angle = rect

    max_dim = max(width, height)
    center = (int(center_x), int(center_y))

    if debug_output_path:
        debug_img = image.copy()
        cv2.drawContours(debug_img, [max_contour], -1, (0, 255, 0), 3)
        cv2.circle(debug_img, center, 5, (0, 0, 255), -1)
        cv2.imwrite(debug_output_path, debug_img)
        print(f"  Debug: contour image saved to: {debug_output_path}")

    return center, max_dim

def rotate_image(image, angle, center_point=None, fill_color=(255, 255, 255)):
    """Rotate an image around a given center point, filling empty areas with fill_color."""
    (h, w) = image.shape[:2]

    if center_point is None:
        (cX, cY) = (w // 2, h // 2)
    else:
        (cX, cY) = center_point

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_color
    )
    return rotated

def crop_to_square(image, center, size, padding_percent, fill_color=(255, 255, 255)):
    """Crop a square region around center; areas outside the source image are filled with fill_color."""
    (h, w) = image.shape[:2]

    target_size = int(size * (1 + padding_percent))
    target_size = max(target_size, 100)
    half_size = target_size // 2

    start_x = int(center[0] - half_size)
    end_x = start_x + target_size
    start_y = int(center[1] - half_size)
    end_y = start_y + target_size

    border_left = max(0, -start_x)
    border_right = max(0, end_x - w)
    border_top = max(0, -start_y)
    border_bottom = max(0, end_y - h)

    if border_left > 0 or border_right > 0 or border_top > 0 or border_bottom > 0:
        image_padded = cv2.copyMakeBorder(
            image,
            top=border_top,
            bottom=border_bottom,
            left=border_left,
            right=border_right,
            borderType=cv2.BORDER_CONSTANT,
            value=fill_color
        )

        start_x += border_left
        end_x += border_left
        start_y += border_top
        end_y += border_top

        cropped_image = image_padded[start_y:end_y, start_x:end_x]
    else:
        cropped_image = image[start_y:end_y, start_x:end_x]

    return cropped_image

def estimate_weighted_background_color(image, center, max_dim, padding_distance_factor=1.5):
    """
    Estimate a distance-weighted average background color (vectorized with NumPy).
    """
    (h, w) = image.shape[:2]
    (cX, cY) = center

    min_dist_to_sample = max_dim * padding_distance_factor / 2

    Y, X = np.ogrid[:h, :w]
    dist_map = np.sqrt((X - cX)**2 + (Y - cY)**2)

    background_mask = dist_map > min_dist_to_sample

    if not np.any(background_mask):
        return (255, 255, 255)

    weights = dist_map[background_mask] - min_dist_to_sample
    background_pixels = image[background_mask]

    weights_expanded = weights[:, np.newaxis]

    weighted_color_sum = np.sum(background_pixels * weights_expanded, axis=0)
    total_weight = np.sum(weights)

    if total_weight > 0:
        avg_color = weighted_color_sum / total_weight
        return tuple(map(int, np.clip(np.round(avg_color), 0, 255)))
    else:
        return (255, 255, 255)

# ======================================================================
# -------------------- 2. PyQt GUI --------------------
# ======================================================================
class CoinProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coin Photo Batch Processor")
        self.setGeometry(100, 100, 1400, 800)

        # Image data and parameter storage
        self.image_files = []
        self.original_images = {}
        self.image_params = {}
        self.current_index = -1

        self.DEFAULT_PARAMS = {
            'gamma': 1.0,
            'contrast': 1.0,
            'brightness': 0,
            'rotation': 0,
            'padding_percent': 0.40
        }

        self.fill_mode = 'white'

        # --- Apply-to-all system ---
        # Each parameter has its own independent "apply to all" toggle, so
        # e.g. rotation can broadcast while padding stays per-image.
        self.APPLY_ALL_KEYS = ['gamma', 'contrast', 'brightness', 'rotation', 'padding_percent']
        self.apply_all_flags = {key: False for key in self.APPLY_ALL_KEYS}

        # Filenames the user has excluded from the apply-to-all mechanism.
        # An excluded image never receives a broadcast from another image,
        # and its own edits never get broadcast to others - it behaves as a
        # fully independent image regardless of the apply-to-all toggles.
        self.excluded_from_apply_all = set()

        # --- Optimization: cache of the last processed full-resolution pixmap ---
        # Resizing the window no longer re-runs the whole OpenCV pipeline; it
        # just rescales this cached QPixmap.
        self._last_pixmap = None

        # --- Optimization: cache of the gamma/contrast/brightness-adjusted
        # image + detected coin center/size, keyed per filename. Rotation and
        # padding changes (the most frequently tweaked controls) no longer
        # re-run gamma correction, contrast/brightness, or contour detection.
        self._detection_cache = {}  # filename -> dict(gamma, contrast, brightness, adjusted, center, size)

        # --- Optimization: debounce slider/spinbox-driven reprocessing so a
        # fast drag doesn't trigger a full pipeline run on every intermediate
        # value; only the final value ~120ms after the user stops moving it.
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(120)
        self._update_timer.timeout.connect(self._recompute_and_display)

        self._setup_ui()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ------------------- A. File list and loading (left) -------------------
        file_panel = QWidget()
        file_layout = QVBoxLayout(file_panel)

        self.list_widget = QListWidget()
        self.list_widget.setMinimumWidth(200)
        self.list_widget.currentItemChanged.connect(self.file_selection_changed)
        self.list_widget.itemChanged.connect(self._handle_item_check_changed)

        file_layout.addWidget(QLabel(
            "Tip: uncheck a photo's box to exclude it from \"apply to all\"."
        ))

        btn_load = QPushButton("Load Image Folder")
        btn_load.clicked.connect(self.load_images)
        file_layout.addWidget(btn_load)

        h_file_buttons = QHBoxLayout()
        btn_add_files = QPushButton("Add Photos")
        btn_add_files.clicked.connect(self.add_images)
        btn_remove_files = QPushButton("Remove Selected")
        btn_remove_files.clicked.connect(self.remove_selected_images)
        h_file_buttons.addWidget(btn_add_files)
        h_file_buttons.addWidget(btn_remove_files)
        file_layout.addLayout(h_file_buttons)

        file_layout.addWidget(QLabel("Image File List:"))
        file_layout.addWidget(self.list_widget)

        # ------------------- B. Image display (center) -------------------
        self.image_label = QLabel("Please load photos")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 600)
        self.image_label.setStyleSheet("border: 1px solid #ddd; background-color: #f0f0f0;")

        # ------------------- C. Control panel (right) -------------------
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        self.control_panel.setMinimumWidth(300)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.control_panel)

        self._setup_controls()

        # ------------------- D. Combined layout -------------------
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(file_panel)
        splitter.addWidget(self.image_label)
        splitter.addWidget(scroll)
        splitter.setSizes([200, 800, 300])

        main_layout.addWidget(splitter)

        # ------------------- E. Global shortcuts (QShortcut) -------------------
        self.shortcut_rot_down = QShortcut(QKeySequence('['), self)
        self.shortcut_rot_down.activated.connect(lambda: self._handle_rotation_shortcut(-5))

        self.shortcut_rot_up = QShortcut(QKeySequence(']'), self)
        self.shortcut_rot_up.activated.connect(lambda: self._handle_rotation_shortcut(5))

        self.shortcut_prev = QShortcut(QKeySequence(Qt.Key_PageUp), self)
        self.shortcut_prev.activated.connect(lambda: self._handle_file_navigation(-1))

        self.shortcut_next = QShortcut(QKeySequence(Qt.Key_PageDown), self)
        self.shortcut_next.activated.connect(lambda: self._handle_file_navigation(1))

    def _handle_rotation_shortcut(self, step):
        """Receive a QShortcut signal and update the rotation angle."""
        if self.current_index == -1:
            return

        current_rotation = self.rotation_spin.value()
        new_rotation = current_rotation + step
        self.rotation_spin.setValue(max(-180, min(180, new_rotation)))

    def _handle_file_navigation(self, direction):
        """Receive a QShortcut signal and switch to the previous/next photo."""
        if self.current_index == -1:
            return

        total_files = self.list_widget.count()
        if total_files <= 1:
            return

        new_index = self.current_index + direction

        if new_index < 0:
            new_index = 0
        elif new_index >= total_files:
            new_index = total_files - 1

        self.list_widget.setCurrentRow(new_index)

    def _setup_controls(self):
        """Build the parameter-adjustment controls on the right."""

        mode_group = QGroupBox("Apply to All (per parameter)")
        mode_layout = QVBoxLayout(mode_group)

        # One independent checkbox per parameter. When checked, changing that
        # parameter on any (non-excluded) photo broadcasts the new value to
        # every other (non-excluded) photo.
        self.apply_all_checks = {}
        param_labels = {
            'gamma': "Exposure",
            'contrast': "Contrast",
            'brightness': "Brightness",
            'rotation': "Rotation",
            'padding_percent': "Padding",
        }
        for key in self.APPLY_ALL_KEYS:
            cb = QCheckBox(param_labels[key])
            cb.setChecked(self.apply_all_flags[key])
            cb.stateChanged.connect(lambda state, k=key: self._set_apply_all_flag(k, state))
            self.apply_all_checks[key] = cb
            mode_layout.addWidget(cb)

        btn_cancel_apply_all = QPushButton("Cancel Apply to All")
        btn_cancel_apply_all.setToolTip(
            "Turn off every 'apply to all' toggle above. Existing per-photo "
            "values are left as they are - only future broadcasting stops."
        )
        btn_cancel_apply_all.clicked.connect(self._cancel_apply_to_all)
        mode_layout.addWidget(btn_cancel_apply_all)

        self.control_layout.addWidget(mode_group)

        fill_group = QGroupBox("Fill Color Option")
        fill_layout = QVBoxLayout(fill_group)

        self.fill_combo = QComboBox()
        self.fill_combo.addItem("White Fill (255, 255, 255)", 'white')
        self.fill_combo.addItem("Black Fill (0, 0, 0)", 'black')
        self.fill_combo.addItem("Weighted Average (Smart Fill)", 'weighted')
        self.fill_combo.setCurrentIndex(0)
        self.fill_combo.currentIndexChanged.connect(self._update_fill_mode)

        fill_layout.addWidget(self.fill_combo)
        self.control_layout.addWidget(fill_group)

        basic_group = QGroupBox("Basic Parameter Adjustment")
        basic_layout = QVBoxLayout(basic_group)
        self.control_layout.addWidget(basic_group)

        self.gamma_layout, self.gamma_slider, self.gamma_input = self._create_slider_control(
            "Exposure (Gamma):", 1.0, 0.1, 3.0, 100, lambda v: self._update_params('gamma', v)
        )
        basic_layout.addLayout(self.gamma_layout)

        self.contrast_layout, self.contrast_slider, self.contrast_input = self._create_slider_control(
            "Contrast (Alpha):", 1.0, 0.1, 3.0, 100, lambda v: self._update_params('contrast', v)
        )
        basic_layout.addLayout(self.contrast_layout)

        self.brightness_layout, self.brightness_slider, self.brightness_input = self._create_slider_control(
            "Brightness (Beta):", 0, -100, 100, 1, lambda v: self._update_params('brightness', v)
        )
        basic_layout.addLayout(self.brightness_layout)

        geo_group = QGroupBox("Geometric Transformations")
        geo_layout = QVBoxLayout(geo_group)
        self.control_layout.addWidget(geo_group)

        rot_layout = QHBoxLayout()
        rot_layout.addWidget(QLabel("Rotation Angle (°):"))
        self.rotation_spin = QSpinBox()
        self.rotation_spin.setRange(-180, 180)
        self.rotation_spin.setSingleStep(5)
        self.rotation_spin.setValue(0)
        self.rotation_spin.valueChanged.connect(lambda v: self._update_params('rotation', v))
        rot_layout.addWidget(self.rotation_spin)
        geo_layout.addLayout(rot_layout)

        pad_layout = QHBoxLayout()
        pad_layout.addWidget(QLabel("Padding Percentage (%):"))
        self.padding_spin = QDoubleSpinBox()
        self.padding_spin.setRange(0, 10000)  # effectively unlimited
        self.padding_spin.setSingleStep(5)
        self.padding_spin.setValue(40)
        self.padding_spin.setSuffix("%")
        self.padding_spin.valueChanged.connect(lambda v: self._update_params('padding_percent', v / 100.0))
        pad_layout.addWidget(self.padding_spin)
        geo_layout.addLayout(pad_layout)

        prefix_group = QGroupBox("Output Naming")
        prefix_layout = QHBoxLayout(prefix_group)
        prefix_layout.addWidget(QLabel("Output Prefix:"))
        self.prefix_input = QLineEdit()
        self.prefix_input.setPlaceholderText("e.g., 'final_' (Empty uses original name)")
        prefix_layout.addWidget(self.prefix_input)
        self.control_layout.addWidget(prefix_group)

        btn_save_current = QPushButton("Save Current Photo")
        btn_save_current.clicked.connect(self.save_current_image)
        self.control_layout.addWidget(btn_save_current)

        btn_save_all = QPushButton("Batch Save All Photos")
        btn_save_all.clicked.connect(self.save_all_images)
        self.control_layout.addWidget(btn_save_all)

        self.control_layout.addStretch(1)

    def _create_slider_control(self, label_text, default_val, min_val, max_val, factor, callback):
        """
        Build a slider + line-edit parameter control pair.
        Returns: (QHBoxLayout, QSlider, QLineEdit)
        """
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel(label_text))

        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(min_val * factor), int(max_val * factor))
        slider.setValue(int(default_val * factor))

        line_edit = QLineEdit(f"{default_val:.2f}" if factor > 1 else str(int(default_val)))
        line_edit.setFixedWidth(50)

        def slider_changed(val):
            actual_val = val / factor
            line_edit.setText(f"{actual_val:.2f}" if factor > 1 else str(int(actual_val)))
            callback(actual_val)

        def input_changed():
            try:
                actual_val = float(line_edit.text())
                actual_val = max(min_val, min(max_val, actual_val))
                slider.setValue(int(actual_val * factor))
                callback(actual_val)
            except ValueError:
                slider_changed(slider.value())

        slider.valueChanged.connect(slider_changed)
        line_edit.editingFinished.connect(input_changed)

        h_layout.addWidget(slider)
        h_layout.addWidget(line_edit)

        return h_layout, slider, line_edit

    def _update_fill_mode(self, index):
        """Update the fill mode and refresh the display."""
        self.fill_mode = self.fill_combo.itemData(index)
        self.update_image_display()

    # ------------------- Apply-to-All Methods -------------------

    def _set_apply_all_flag(self, key, state):
        """Toggle whether a given parameter broadcasts to all (non-excluded) photos."""
        self.apply_all_flags[key] = (state == Qt.Checked)

    def _cancel_apply_to_all(self):
        """
        Turn off every 'apply to all' toggle at once.
        This only stops future broadcasting - it does not touch any
        parameter values already set on individual photos, and it does not
        change per-photo exclusion state.
        """
        for key in self.APPLY_ALL_KEYS:
            self.apply_all_flags[key] = False
            checkbox = self.apply_all_checks.get(key)
            if checkbox is not None:
                checkbox.blockSignals(True)
                checkbox.setChecked(False)
                checkbox.blockSignals(False)
        print("Apply-to-all cancelled for all parameters.")

    def _handle_item_check_changed(self, item):
        """
        Track which photos are excluded from the apply-to-all mechanism via
        the checkbox next to each filename in the list. Unchecked = excluded:
        the photo neither sends nor receives apply-to-all broadcasts.
        """
        filename = item.text()
        if item.checkState() == Qt.Checked:
            self.excluded_from_apply_all.discard(filename)
        else:
            self.excluded_from_apply_all.add(filename)

    # ------------------- File/Data Management Methods -------------------

    def load_images(self):
        """Open a dialog and load all supported image files in the chosen folder (non-recursive)."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")

        if dir_path:
            self.image_files = []
            self.original_images = {}
            self.image_params = {}
            self._detection_cache = {}
            self.list_widget.clear()

            # Match both lower- and upper-case extensions. On Windows the
            # filesystem is case-insensitive so this was harmless there, but
            # on Linux/macOS a lowercase-only glob silently skips files like
            # "Coin1.JPG".
            base_exts = ['jpg', 'jpeg', 'png', 'bmp']
            exts = set(base_exts) | {e.upper() for e in base_exts}
            patterns = [os.path.join(dir_path, f"*.{ext}") for ext in exts]

            all_files = []
            for pattern in patterns:
                all_files.extend(glob.glob(pattern, recursive=False))
            all_files = sorted(set(all_files))

            if not all_files:
                self.image_label.setText("No supported image files found.")
                return

            self._add_files_to_data(all_files)

            if self.image_files:
                self.list_widget.setCurrentRow(0)

    def add_images(self):
        """Add one or more new photos to the list."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Additional Photos", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )

        if file_paths:
            self._add_files_to_data(file_paths)

    def _add_files_to_data(self, file_paths):
        """Internal helper: add new files into the data structures and the list widget."""
        current_paths = set(self.image_files)
        added_count = 0

        for filepath in file_paths:
            if filepath in current_paths:
                print(f"Warning: File already loaded: {os.path.basename(filepath)}")
                continue

            img = cv2.imread(filepath)
            if img is not None:
                filename = os.path.basename(filepath)
                self.image_files.append(filepath)
                self.original_images[filename] = img
                self.image_params[filename] = self.DEFAULT_PARAMS.copy()

                item = QListWidgetItem(filename)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)  # checked = included in "apply to all"
                self.list_widget.addItem(item)

                current_paths.add(filepath)
                added_count += 1
            else:
                print(f"Error loading file: {filepath}")

        if added_count > 0:
            print(f"Successfully added {added_count} new photo(s).")
            if self.current_index == -1 and self.list_widget.count() > 0:
                self.list_widget.setCurrentRow(0)

    def remove_selected_images(self):
        """Remove the selected photos and their associated data from the list."""
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            return

        current_row_index = self.list_widget.currentRow()

        for item in selected_items:
            filename = item.text()
            row = self.list_widget.row(item)

            filepath_to_remove = next((fp for fp in self.image_files if os.path.basename(fp) == filename), None)

            if filepath_to_remove and filepath_to_remove in self.image_files:
                self.image_files.remove(filepath_to_remove)

            if filename in self.original_images:
                del self.original_images[filename]
                del self.image_params[filename]
            self._detection_cache.pop(filename, None)
            self.excluded_from_apply_all.discard(filename)

            self.list_widget.takeItem(row)

        print(f"Removed {len(selected_items)} photo(s).")

        if self.list_widget.count() > 0:
            new_row = min(current_row_index, self.list_widget.count() - 1)
            self.list_widget.setCurrentRow(new_row)
        else:
            self.current_index = -1
            self._last_pixmap = None
            self.image_label.setText("No image loaded or selected.")

    def file_selection_changed(self, current, previous):
        """Triggered when the user selects a different file in the list."""
        if current:
            self.current_index = self.list_widget.currentRow()
            filename = current.text()
            self._load_image_params_to_controls(filename)
            self.update_image_display()

    def _load_image_params_to_controls(self, filename):
        """Load this file's stored parameters into the right-hand controls."""
        params = self.image_params.get(filename, self.DEFAULT_PARAMS)

        self.gamma_slider.setValue(int(params['gamma'] * 100))
        self.contrast_slider.setValue(int(params['contrast'] * 100))
        self.brightness_slider.setValue(int(params['brightness'] * 1))

        self.gamma_input.setText(f"{params['gamma']:.2f}")
        self.contrast_input.setText(f"{params['contrast']:.2f}")
        self.brightness_input.setText(str(int(params['brightness'])))

        self.rotation_spin.setValue(int(params['rotation']))
        self.padding_spin.setValue(int(params['padding_percent'] * 100))

    def _update_params(self, key, value):
        """Update the parameter for the current file (or all files) and refresh the display."""
        if self.current_index == -1 or self.list_widget.currentItem() is None:
            return

        current_filename = self.list_widget.currentItem().text()

        # Always set the value on the photo being edited.
        self.image_params[current_filename][key] = value

        # Broadcast to the other photos only if:
        #  1) this specific parameter's "apply to all" toggle is on, and
        #  2) the photo being edited is not itself excluded (an excluded
        #     photo's edits stay local to itself and never go out).
        if self.apply_all_flags.get(key) and current_filename not in self.excluded_from_apply_all:
            for filename in self.image_params:
                if filename == current_filename:
                    continue
                # Excluded photos never receive broadcasts either.
                if filename in self.excluded_from_apply_all:
                    continue
                self.image_params[filename][key] = value

        # --- Optimization: debounce. Sliders can fire many valueChanged
        # events per second while dragging; only the settled value (120ms
        # after the last change) triggers a real recompute.
        self._update_timer.start()

    # ------------------- Image Processing and Display Methods -------------------

    def cv2_to_qpixmap(self, cv_img):
        """Convert an OpenCV BGR image (numpy array) to a QPixmap."""
        if cv_img is None:
            return QPixmap()

        cv_img = np.ascontiguousarray(cv_img)

        if len(cv_img.shape) == 3:
            h, w, ch = cv_img.shape
            bytes_per_line = ch * w
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            q_format = QImage.Format_RGB888
        else:
            h, w = cv_img.shape
            bytes_per_line = w
            rgb_image = cv_img
            q_format = QImage.Format_Grayscale8

        q_image = QImage(rgb_image.data, w, h, bytes_per_line, q_format)
        # .copy() detaches the QImage from the numpy buffer, which may be
        # reused/garbage-collected once this function returns.
        return QPixmap.fromImage(q_image.copy())

    def process_image_pipeline(self, filename):
        """
        Full image-processing pipeline.

        Optimization: gamma/contrast/brightness adjustment and coin-contour
        detection are the most expensive steps. They only depend on
        (gamma, contrast, brightness), not on rotation or padding. We cache
        their result per filename and reuse it whenever those three values
        haven't changed since the last run - this makes rotation/padding
        tweaks (by far the most common interactive adjustment) essentially free
        aside from the warpAffine/crop calls.
        """
        original_img = self.original_images.get(filename)
        params = self.image_params.get(filename)

        if original_img is None or params is None:
            return None

        gamma = params['gamma']
        contrast = params['contrast']
        brightness = params['brightness']

        cache_entry = self._detection_cache.get(filename)
        if (cache_entry is not None
                and cache_entry['gamma'] == gamma
                and cache_entry['contrast'] == contrast
                and cache_entry['brightness'] == brightness):
            img_adjusted = cache_entry['adjusted']
            coin_center = cache_entry['center']
            size = cache_entry['size']
        else:
            img_adjusted = adjust_gamma(original_img, gamma)
            img_adjusted = adjust_alpha_beta(img_adjusted, contrast, brightness)
            coin_center, size = find_coin_center_and_size(img_adjusted)

            self._detection_cache[filename] = {
                'gamma': gamma,
                'contrast': contrast,
                'brightness': brightness,
                'adjusted': img_adjusted,
                'center': coin_center,
                'size': size,
            }

        rotation_angle = params['rotation']
        padding_percent = params['padding_percent']

        fill_color_bgr = (255, 255, 255)
        if self.fill_mode == 'black':
            fill_color_bgr = (0, 0, 0)
        elif self.fill_mode == 'white':
            fill_color_bgr = (255, 255, 255)
        elif self.fill_mode == 'weighted':
            # Only run the (relatively costly) weighted background estimate
            # when rotation actually introduces empty corners to fill.
            if abs(rotation_angle) > 0:
                fill_color_bgr = estimate_weighted_background_color(
                    img_adjusted, coin_center, size
                )

        img_rotated = rotate_image(
            img_adjusted, rotation_angle, center_point=coin_center, fill_color=fill_color_bgr
        )

        final_img = crop_to_square(
            img_rotated, coin_center, size, padding_percent, fill_color=fill_color_bgr
        )

        return final_img

    def _recompute_and_display(self):
        """Run the (debounced) full pipeline and refresh the display."""
        if self.current_index == -1 or self.list_widget.currentItem() is None:
            self._last_pixmap = None
            self.image_label.setText("No image loaded or selected.")
            return

        current_filename = self.list_widget.currentItem().text()
        processed_img = self.process_image_pipeline(current_filename)

        if processed_img is not None:
            self._last_pixmap = self.cv2_to_qpixmap(processed_img)
            self._render_cached_pixmap()
        else:
            self._last_pixmap = None
            self.image_label.setText(f"Processing failed: {current_filename}")

    def _render_cached_pixmap(self):
        """Scale the cached full-resolution pixmap to fit the label - no OpenCV work involved."""
        if self._last_pixmap is None:
            return
        scaled_pixmap = self._last_pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setText("")

    def update_image_display(self):
        """Immediately (non-debounced) reprocess and redisplay the current image."""
        self._update_timer.stop()
        self._recompute_and_display()

    def resizeEvent(self, event):
        """
        On window resize, just rescale the already-cached pixmap instead of
        re-running the full OpenCV pipeline (gamma/contrast/contour
        detection/rotate/crop) again. This is the single biggest win for UI
        responsiveness, since resize events fire continuously while dragging
        the window edge.
        """
        super().resizeEvent(event)
        self._render_cached_pixmap()

    def keyPressEvent(self, event):
        """Handle keyboard events not already captured by QShortcut."""
        return super().keyPressEvent(event)

    # ------------------- Save Methods -------------------

    def save_current_image(self):
        """Save the currently displayed photo."""
        if self.current_index == -1 or self.list_widget.currentItem() is None:
            return

        current_filename = self.list_widget.currentItem().text()
        processed_img = self.process_image_pipeline(current_filename)

        if processed_img is not None:
            prefix = self.prefix_input.text().strip()
            name, ext = os.path.splitext(current_filename)

            default_name = f"{prefix}{name}{ext}" if prefix else current_filename

            filepath, _ = QFileDialog.getSaveFileName(
                self, "Save Processed Photo", default_name, "Images (*.jpg *.png)"
            )
            if filepath:
                success = cv2.imwrite(filepath, processed_img)
                if success:
                    print(f"Successfully saved current file: {filepath}")
                    QMessageBox.information(self, "Success", f"Photo successfully saved to:\n{filepath}")
                else:
                    print(f"Error saving file: {filepath}")
                    QMessageBox.critical(self, "Error", f"Failed to save photo to:\n{filepath}")

    def save_all_images(self):
        """Process and save all photos in a batch."""
        if not self.image_files:
            return

        save_dir = QFileDialog.getExistingDirectory(self, "Select Batch Output Folder")
        if not save_dir:
            return

        prefix = self.prefix_input.text().strip()
        count = 0

        for filepath in self.image_files:
            filename = os.path.basename(filepath)
            processed_img = self.process_image_pipeline(filename)

            if processed_img is not None:
                name, ext = os.path.splitext(filename)
                output_filename = f"{prefix}{name}{ext}" if prefix else filename
                output_filepath = os.path.join(save_dir, output_filename)

                success = cv2.imwrite(output_filepath, processed_img)
                if success:
                    count += 1
                else:
                    print(f"Warning: Failed to save file {filename} to {output_filepath}")

        print("--- Batch Save Complete ---")
        print(f"Successfully processed and saved {count} photos to {save_dir}")

        QMessageBox.information(self, "Batch Complete",
                                f"Batch processing finished.\nSuccessfully saved {count} photos to:\n{save_dir}")

if __name__ == '__main__':
    app = QApplication([])
    window = CoinProcessorApp()
    window.show()
    app.exec_()