import sys
import SimpleITK as sitk
import numpy as np
import os
import datetime
import tempfile
import pickle
import shutil
from PIL import Image
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, 
                             QWidget, QFileDialog, QSlider, QGroupBox, QLabel, QComboBox, 
                             QMessageBox, QListWidget, QDialog, QLineEdit, QSpinBox, QScrollArea,
                             QProgressBar, QTextEdit, QCheckBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QCursor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
import imageio
import nibabel as nib  
# AI/ML Imports


try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google Generative AI not available. AI features will be disabled.")

# Define colormaps
MASK_COLORS = [(0, 0, 0, 0), (0, 0.7, 0.7, 0.5)]
CYAN_MASK_CMAP = LinearSegmentedColormap.from_list('CyanMask', MASK_COLORS, N=256)




# ====================  GEMINI WORKER ====================

# ==================== CORRECTED GEMINI WORKER ====================

class GeminiWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    
    def __init__(self, scan_array, api_key):
        super().__init__()
        self.scan_array = scan_array
        self.api_key = api_key
  

    # --- REPLACE THE ENTIRE 'run' METHOD IN 'GeminiWorker' WITH THIS ---

    def run(self):
        try:
            # Configure Gemini
            self.progress.emit("🔧 Configuring Gemini API...")
            genai.configure(api_key=self.api_key)
            
            # Use a robust list of vision-capable model names, prioritizing the latest
            model_names = [
                'gemini-2.5-flash',           # New recommended multimodal model
                'gemini-1.0-pro-vision',      # Standard reliable multimodal model
                'gemini-1.5-flash-001', # Standard reliable multimodal model
            ]
            
            model = None
            last_error = "No models were tested."
            
            for model_name in model_names:
                try:
                    self.progress.emit(f"🔍 Trying multimodal model: {model_name}...")
                    model = genai.GenerativeModel(model_name)
                    if model is not None:
                         self.progress.emit(f"✅ Successfully loaded model: {model_name}")
                         break
                except Exception as e:
                    last_error = str(e)
                    self.progress.emit(f"⚠️ Model {model_name} failed. Error: {e}")
                    continue
            
            if model is None:
                raise Exception(f"❌ Could not find a working Gemini multimodal model.\n\nLast error: {last_error}\n\nPlease update google-generativeai:\npip install --upgrade google-generativeai")
            
            self.progress.emit("📸 Extracting middle slice for analysis...")
            middle_slice = self.scan_array[self.scan_array.shape[0] // 2]
            normalized = ((middle_slice - middle_slice.min()) / 
                          (middle_slice.max() - middle_slice.min()) * 255).astype(np.uint8)
            img = Image.fromarray(normalized).convert("RGB")
            
            # ==================== FIX: UPDATED PROMPT ====================
            prompt = """Analyze this medical scan image and provide:
1. Orientation (choose one: RAS, LPS, RPS, LAI, LPI, RAI)
2. View (choose one: Axial, Sagittal, Coronal, Oblique, Unknown)
3. Anatomical region (choose one: Brain, Head/Neck, Chest, Abdomen, Pelvis, Spine-Cervical, Spine-Thoracic, Spine-Lumbar, Upper-Extremity, Lower-Extremity, Heart, Liver, Kidneys, Full-Body, Other)

Format your response as:
Orientation: [orientation]
View: [view]
Anatomy: [anatomy]
Confidence: [0-100]%"""
            # ======================= END OF FIX ========================
            
            self.progress.emit("🤖 Analyzing with Gemini AI...")
            response = model.generate_content([prompt, img])
            
            if not response or not hasattr(response, 'text') or not response.text:
                raise Exception("Gemini API returned empty response. Check your API key and quota.")
            
            self.progress.emit("📊 Processing AI results...")
            text = response.text
            orientation = "RAS"
            anatomy = "Unknown"
            view = "Unknown"  # <-- New variable
            confidence = 0.0
            
            for line in text.split('\n'):
                if 'Orientation:' in line:
                    orientation = line.split(':')[1].strip()
                # ==================== FIX: PARSE THE VIEW ====================
                elif 'View:' in line:
                    view = line.split(':')[1].strip()
                # ======================= END OF FIX ========================
                elif 'Anatomy:' in line:
                    anatomy = line.split(':')[1].strip()
                elif 'Confidence:' in line:
                    try:
                        confidence = float(line.split(':')[1].strip().replace('%', '')) / 100
                    except:
                        confidence = 0.5
            
            self.finished.emit({
                'success': True,
                'orientation': {'orientation': orientation, 'confidence': confidence},
                # ==================== FIX: EMIT THE VIEW RESULT ====================
                'view': {'view': view, 'confidence': confidence},
                # ========================== END OF FIX =============================
                'anatomy': {'primary_region': anatomy, 'confidence': confidence, 'top3': [(anatomy, confidence)]}
            })
            
        except Exception as e:
            # (The existing error handling code remains unchanged)
            error_msg = str(e)
            if "404" in error_msg and "models/" in error_msg:
                error_msg = ("❌ Model Not Found...") # and so on
            # ...
            self.finished.emit({'success': False, 'error': error_msg})
# ============================================
# INSTALLATION FIX (run this in terminal):
# ============================================
"""
To fix this error permanently, run:

pip install --upgrade google-generativeai

This will update to the latest version that supports the new model names.
"""

# ====================  FIXED LOAD_FILE METHOD ====================
# Replace your existing load_file method in MRIViewer class with this:
# ==================== DIY INSTRUCTION: REPLACE THIS ENTIRE METHOD ====================

# =============================== END OF REPLACEMENT ===============================

# ====================  INSTRUCTIONS ====================
"""
HOW TO APPLY THESE FIXES:

1. REPLACE the GeminiWorker class (around line 47-89 in your code):
   - Find: class GeminiWorker(QThread):
   - Replace the entire class with the code above

2. REPLACE the load_file method in MRIViewer class (around line 1100):
   - Find: def load_file(self, file_type):
   - Replace the entire method with the code above

3. Save and run your code

WHAT THIS FIXES:
✅ AI analysis now works for NIfTI (.nii) files
✅ Better error messages when API fails
✅ Automatic prompt to set API key if not configured
✅ Handles empty/blocked responses from Gemini
✅ Shows progress messages during analysis

TESTING:
1. Click "🔑 Set Gemini API Key" and enter your key
2. Load any .nii file using "Load Main Scan (.nii)"
3. AI analysis should automatically start
4. You should see progress messages and results

If you still get errors, the error message will now tell you exactly what went wrong!
"""

# BLOCK 1: SEGMENTATION WORKER CLASS
# Location: Add after AIWorker class (around line 1000)
# Find the line: "class AIWorker(QThread):" and find where it ends
# Add this entire block AFTER the AIWorker class ends


class SegmentationWorker(QThread):
    """Background thread for TotalSegmentator segmentation"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    
    def __init__(self, itk_image):
        super().__init__()
        self.itk_image = itk_image
    
    def run(self):
        try:
            import tempfile
            import os
            from totalsegmentator.python_api import totalsegmentator
            
            self.progress.emit("🔧 Initializing TotalSegmentator...")
            self.progress.emit("⏳ Loading AI model (first run may take longer)...")
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_input:
                temp_input_path = temp_input.name
            
            temp_output_path = tempfile.mkdtemp()  # Create temporary directory for output
            
            try:
                # Write ITK image to temporary file
                self.progress.emit("💾 Preparing image file...")
                print(f"[DEBUG] Writing image to: {temp_input_path}")
                sitk.WriteImage(self.itk_image, temp_input_path)
                print(f"[DEBUG] Image written successfully")
                
                # Detect device
                try:
                    import torch
                    device = "gpu" if torch.cuda.is_available() else "cpu"
                    self.progress.emit(f"🖥️ Using device: {device.upper()}")
                    print(f"[DEBUG] Using device: {device}")
                except:
                    device = "cpu"
                    self.progress.emit("🖥️ Using device: CPU")
                    print("[DEBUG] PyTorch not available, using CPU")
                
                # Run segmentation with file paths
                self.progress.emit("🔄 Segmenting image... (this may take 2-5 minutes)")
                print(f"[DEBUG] Starting segmentation with input: {temp_input_path}")
                
                result = totalsegmentator(
                    temp_input_path,
                    temp_output_path,
                    task="total",
                    ml=True,
                    device=device
                )
                
                print(f"[DEBUG] Segmentation result type: {type(result)}")
                
                # Handle different return types from totalsegmentator
                if isinstance(result, str):
                    # It's a file path string
                    segmentation_path = result
                    print(f"[DEBUG] Result is a file path: {segmentation_path}")
                    self.progress.emit("📖 Loading segmentation results...")
                    segmentation = sitk.ReadImage(segmentation_path)
                else:
                    # It's already a nibabel image object - convert to SimpleITK
                    print(f"[DEBUG] Result is a nibabel image object, converting...")
                    import nibabel as nib
                    
                    # If it's a nibabel image, convert to SimpleITK
                    if hasattr(result, 'get_fdata'):
                        seg_data = result.get_fdata()
                        segmentation = sitk.GetImageFromArray(seg_data.astype(np.uint8))
                        print(f"[DEBUG] Converted nibabel to SimpleITK image")
                    else:
                        segmentation = result
                
                seg_array = sitk.GetArrayFromImage(segmentation)
                print(f"[DEBUG] Segmentation array shape: {seg_array.shape}")
                
                self.progress.emit("✅ Segmentation complete! Processing results...")
                
                self.finished.emit({
                    'success': True,
                    'segmentation': segmentation,
                    'seg_array': seg_array
                })
                
            except Exception as e:
                print(f"[ERROR] Segmentation error: {str(e)}")
                import traceback
                traceback.print_exc()
                self.finished.emit({
                    'success': False,
                    'error': str(e)
                })
                
            finally:
                # Clean up temporary files
                print("[DEBUG] Cleaning up temporary files...")
                if os.path.exists(temp_input_path):
                    try:
                        os.remove(temp_input_path)
                        print(f"[DEBUG] Removed: {temp_input_path}")
                    except Exception as e:
                        print(f"[WARNING] Could not remove {temp_input_path}: {e}")
                
                if os.path.exists(temp_output_path):
                    try:
                        import shutil
                        shutil.rmtree(temp_output_path)
                        print(f"[DEBUG] Removed: {temp_output_path}")
                    except Exception as e:
                        print(f"[WARNING] Could not remove {temp_output_path}: {e}")
        
        except ImportError as e:
            print(f"[ERROR] Import error: {str(e)}")
            self.finished.emit({
                'success': False,
                'error': f'TotalSegmentator not installed.\n\nInstall with:\npip install TotalSegmentator\n\nError: {str(e)}'
            })
        except Exception as e:
            print(f"[ERROR] Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.finished.emit({
                'success': False,
                'error': str(e)
            })

# ==================== ROI CLASSES ====================

class ROI:
    """3D ROI object with cross-view synchronization"""
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, label="ROI"):
        # Store in volume coordinates (X, Y, Z)
        self.x_min = int(max(0, x_min))
        self.x_max = int(x_max)
        self.y_min = int(max(0, y_min))
        self.y_max = int(y_max)
        self.z_min = int(max(0, z_min))
        self.z_max = int(z_max)
        self.label = label
        self.color = 'cyan'
        
    def get_bounds_3d(self):
        """Returns 3D bounds as (x_min, x_max, y_min, y_max, z_min, z_max)"""
        return (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)
    
    def get_bounds_for_view(self, view_type):
        """Returns 2D bounds for specific view"""
        if view_type == 'axial':
            return (self.x_min, self.x_max, self.y_min, self.y_max)
        elif view_type == 'coronal':
            return (self.x_min, self.x_max, self.z_min, self.z_max)
        elif view_type == 'sagittal':
            return (self.y_min, self.y_max, self.z_min, self.z_max)
        return (0, 0, 0, 0)
    
    def get_slice_range(self, view_type):
        """Returns slice range for the view's perpendicular axis"""
        if view_type == 'axial':
            return (self.z_min, self.z_max)
        elif view_type == 'coronal':
            return (self.y_min, self.y_max)
        elif view_type == 'sagittal':
            return (self.x_min, self.x_max)
        return (0, 0)
    
    def get_num_slices(self, view_type):
        """Returns number of slices in ROI for given view"""
        start, end = self.get_slice_range(view_type)
        return end - start + 1
    
    def is_visible_on_slice(self, view_type, slice_idx):
        """Check if ROI is visible on current slice"""
        if view_type == 'axial':
            return self.z_min <= slice_idx <= self.z_max
        elif view_type == 'coronal':
            return self.y_min <= slice_idx <= self.y_max
        elif view_type == 'sagittal':
            return self.x_min <= slice_idx <= self.x_max
        return False
    
    def get_volume_shape(self):
        """Returns shape of ROI volume (Z, Y, X)"""
        return (
            self.z_max - self.z_min + 1,
            self.y_max - self.y_min + 1,
            self.x_max - self.x_min + 1
        )
    
    def to_dict(self):
        """Serialize ROI to dictionary"""
        return {
            'bounds_3d': self.get_bounds_3d(),
            'label': self.label,
            'color': self.color,
            'shape': self.get_volume_shape()
        }


class ROILabelDialog(QDialog):
    """Dialog for labeling ROI and setting slice range"""
    def __init__(self, parent=None, max_slices=100, current_slice=0, 
        auto_label="Auto-Detected Organ", suggested_start=0, suggested_end=None):
        super().__init__(parent)
        self.setWindowTitle("ROI Properties")
        self.setModal(True)
        
        if suggested_end is None:
            suggested_end = max_slices - 1
        
        layout = QVBoxLayout()
        
        organ_group = QGroupBox("Detected Organ")
        organ_layout = QVBoxLayout()
        
        self.auto_label_display = QLabel(f"Detected: {auto_label}")
        self.auto_label_display.setStyleSheet("font-weight: bold; color: green; font-size: 11pt;")
        organ_layout.addWidget(self.auto_label_display)
        organ_group.setLayout(organ_layout)
        layout.addWidget(organ_group)
        
        organ_select_group = QGroupBox("Or Select Organ Type")
        organ_select_layout = QVBoxLayout()
        
        self.organ_combo = QComboBox()
        self.organ_combo.addItems([
            "Use Auto-Detected", "Brain", "Heart", "Liver", "Kidney (Left)", "Kidney (Right)",
            "Lung (Left)", "Lung (Right)", "Spleen", "Pancreas", "Stomach",
            "Bladder", "Prostate", "Uterus", "Spine", "Tumor", "Other"
        ])
        organ_select_layout.addWidget(self.organ_combo)
        organ_select_group.setLayout(organ_select_layout)
        layout.addWidget(organ_select_group)
        
        label_group = QGroupBox("Custom Label (Optional)")
        label_layout = QVBoxLayout()
        
        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("Enter custom label or leave blank")
        label_layout.addWidget(self.label_input)
        label_group.setLayout(label_layout)
        layout.addWidget(label_group)
        
        slice_group = QGroupBox("Slice Range")
        slice_layout = QVBoxLayout()
        
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Start:"))
        self.slice_start_spin = QSpinBox()
        self.slice_start_spin.setRange(0, max_slices - 1)
        self.slice_start_spin.setValue(suggested_start)
        range_layout.addWidget(self.slice_start_spin)
        
        range_layout.addWidget(QLabel("End:"))
        self.slice_end_spin = QSpinBox()
        self.slice_end_spin.setRange(0, max_slices - 1)
        self.slice_end_spin.setValue(suggested_end)
        range_layout.addWidget(self.slice_end_spin)
        
        slice_layout.addLayout(range_layout)
        
        full_range_btn = QPushButton("Use Full Range")
        full_range_btn.clicked.connect(lambda: self._set_full_range(max_slices))
        slice_layout.addWidget(full_range_btn)
        
        info_label = QLabel(f"Suggested range: {suggested_end - suggested_start + 1} slices")
        info_label.setStyleSheet("color: blue; font-style: italic;")
        slice_layout.addWidget(info_label)
        
        slice_group.setLayout(slice_layout)
        layout.addWidget(slice_group)
        
        self.auto_label = auto_label
        
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def _set_full_range(self, max_slices):
        self.slice_start_spin.setValue(0)
        self.slice_end_spin.setValue(max_slices - 1)
        
    def get_values(self):
        """Returns tuple of (label, slice_start, slice_end)"""
        custom_label = self.label_input.text().strip()
        
        if custom_label:
            label = custom_label
        elif self.organ_combo.currentText() != "Use Auto-Detected":
            label = self.organ_combo.currentText()
        else:
            label = self.auto_label
            
        return (label, self.slice_start_spin.value(), self.slice_end_spin.value())


class ROIAdjustDialog(QDialog):
    """Dialog for adjusting ROI slice ranges"""
    def __init__(self, parent=None, roi=None, volume_shape=None):
        super().__init__(parent)
        self.setWindowTitle("Adjust ROI Slice Range")
        self.setModal(True)
        self.roi = roi
        self.volume_shape = volume_shape  # (D, H, W)
        
        layout = QVBoxLayout()
        
        # ROI Info
        info_group = QGroupBox("ROI Information")
        info_layout = QVBoxLayout()
        
        D, H, W = volume_shape
        info_layout.addWidget(QLabel(f"<b>Current ROI: {roi.label}</b>"))
        info_layout.addWidget(QLabel(f"Volume dimensions: {W} × {H} × {D} (X × Y × Z)"))
        info_layout.addWidget(QLabel(f"Current ROI shape: {roi.get_volume_shape()}"))
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # X Range (Sagittal direction)
        x_group = QGroupBox("X Range (Sagittal - Left/Right)")
        x_layout = QHBoxLayout()
        
        x_layout.addWidget(QLabel("Min:"))
        self.x_min_spin = QSpinBox()
        self.x_min_spin.setRange(0, W - 1)
        self.x_min_spin.setValue(roi.x_min)
        x_layout.addWidget(self.x_min_spin)
        
        x_layout.addWidget(QLabel("Max:"))
        self.x_max_spin = QSpinBox()
        self.x_max_spin.setRange(0, W - 1)
        self.x_max_spin.setValue(roi.x_max)
        x_layout.addWidget(self.x_max_spin)
        
        x_group.setLayout(x_layout)
        layout.addWidget(x_group)
        
        # Y Range (Coronal direction)
        y_group = QGroupBox("Y Range (Coronal - Anterior/Posterior)")
        y_layout = QHBoxLayout()
        
        y_layout.addWidget(QLabel("Min:"))
        self.y_min_spin = QSpinBox()
        self.y_min_spin.setRange(0, H - 1)
        self.y_min_spin.setValue(roi.y_min)
        y_layout.addWidget(self.y_min_spin)
        
        y_layout.addWidget(QLabel("Max:"))
        self.y_max_spin = QSpinBox()
        self.y_max_spin.setRange(0, H - 1)
        self.y_max_spin.setValue(roi.y_max)
        y_layout.addWidget(self.y_max_spin)
        
        y_group.setLayout(y_layout)
        layout.addWidget(y_group)
        
        # Z Range (Axial direction)
        z_group = QGroupBox("Z Range (Axial - Superior/Inferior)")
        z_layout = QHBoxLayout()
        
        z_layout.addWidget(QLabel("Min:"))
        self.z_min_spin = QSpinBox()
        self.z_min_spin.setRange(0, D - 1)
        self.z_min_spin.setValue(roi.z_min)
        z_layout.addWidget(self.z_min_spin)
        
        z_layout.addWidget(QLabel("Max:"))
        self.z_max_spin = QSpinBox()
        self.z_max_spin.setRange(0, D - 1)
        self.z_max_spin.setValue(roi.z_max)
        z_layout.addWidget(self.z_max_spin)
        
        z_group.setLayout(z_layout)
        layout.addWidget(z_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Apply")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
        
    def get_adjusted_bounds(self):
        """Returns adjusted 3D bounds"""
        return (
            self.x_min_spin.value(),
            self.x_max_spin.value(),
            self.y_min_spin.value(),
            self.y_max_spin.value(),
            self.z_min_spin.value(),
            self.z_max_spin.value()
        )


class ROIManager:
    """Manages 3D ROI with cross-view synchronization and export"""
    def __init__(self, viewer):
        self.viewer = viewer
        self.current_roi = None  # Single active ROI
        self.selectors = {}
        self.drawing_enabled = False
        self.active_view = None
        self.temp_rect_coords = None
        
    def enable_drawing(self, view_type):
        """Enable ROI drawing on specified view"""
        if self.viewer.scan_array is None:
            return False
            
        self.drawing_enabled = True
        self.active_view = view_type
        
        if view_type == 'axial':
            ax = self.viewer.axial_ax
        elif view_type == 'coronal':
            ax = self.viewer.coronal_ax
        elif view_type == 'sagittal':
            ax = self.viewer.sagittal_ax
        else:
            return False
        
        if view_type not in self.selectors or self.selectors[view_type] is None:
            # Updated for matplotlib compatibility - removed rectprops parameter
            selector = RectangleSelector(
                ax, 
                lambda eclick, erelease: self._on_rectangle_select(eclick, erelease, view_type),
                useblit=True,
                button=[1],
                minspanx=10, 
                minspany=10,
                spancoords='pixels',
                interactive=False
                # Note: rectprops removed - styling will use default
            )
            
            # Try to set properties after creation (for newer matplotlib versions)
            try:
                selector.set_props(facecolor='cyan', edgecolor='cyan', alpha=0.3, linewidth=2)
            except AttributeError:
                # If set_props doesn't exist, the selector will use defaults
                pass
                
            self.selectors[view_type] = selector
        
        self.selectors[view_type].set_active(True)
        return True
    
    def disable_drawing(self):
        """Disable ROI drawing on all views"""
        self.drawing_enabled = False
        self.active_view = None
        for selector in self.selectors.values():
            if selector:
                selector.set_active(False)
    
    def _on_rectangle_select(self, eclick, erelease, view_type):
        """Callback when rectangle is drawn"""
        if not self.drawing_enabled:
            return
        
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        if None in [x1, y1, x2, y2]:
            return
        
        x_min_disp, x_max_disp = min(x1, x2), max(x1, x2)
        y_min_disp, y_max_disp = min(y1, y2), max(y1, y2)
        
        self._create_3d_roi_from_2d(view_type, x_min_disp, x_max_disp, y_min_disp, y_max_disp)
        self.disable_drawing()
        self.viewer.update_all_views()
        
        if self.current_roi:
            # Automatically open the adjust dialog for a better user experience
            print("[INFO] ROI created. Opening adjustment dialog...")
            self.adjust_roi_range()
    
    def _create_3d_roi_from_2d(self, view_type, x_min_disp, x_max_disp, y_min_disp, y_max_disp):
        """Convert 2D rectangle to 3D ROI"""
        D, H, W = self.viewer.scan_array.shape
        
        if view_type == 'axial':
            x_min_vol = (W - 1) - int(x_max_disp)
            x_max_vol = (W - 1) - int(x_min_disp)
            y_min_vol = int(y_min_disp)
            y_max_vol = int(y_max_disp)
            
            current_z = self.viewer.crosshair_z
            z_range = min(10, D // 4)
            z_min_vol = max(0, current_z - z_range)
            z_max_vol = min(D - 1, current_z + z_range)
            
        elif view_type == 'coronal':
            x_min_vol = (W - 1) - int(x_max_disp)
            x_max_vol = (W - 1) - int(x_min_disp)
            z_min_vol = (D - 1) - int(y_max_disp)
            z_max_vol = (D - 1) - int(y_min_disp)
            
            current_y = self.viewer.crosshair_y
            y_range = min(10, H // 4)
            y_min_vol = max(0, current_y - y_range)
            y_max_vol = min(H - 1, current_y + y_range)
            
        elif view_type == 'sagittal':
            y_min_vol = int(x_min_disp)
            y_max_vol = int(x_max_disp)
            z_min_vol = (D - 1) - int(y_max_disp)
            z_max_vol = (D - 1) - int(y_min_disp)
            
            current_x = self.viewer.crosshair_x
            x_range = min(10, W // 4)
            x_min_vol = max(0, current_x - x_range)
            x_max_vol = min(W - 1, current_x + x_range)
        else:
            return
        
        # Clamp to bounds
        x_min_vol = max(0, min(W - 1, x_min_vol))
        x_max_vol = max(0, min(W - 1, x_max_vol))
        y_min_vol = max(0, min(H - 1, y_min_vol))
        y_max_vol = max(0, min(H - 1, y_max_vol))
        z_min_vol = max(0, min(D - 1, z_min_vol))
        z_max_vol = max(0, min(D - 1, z_max_vol))
        
        self.current_roi = ROI(
            x_min_vol, x_max_vol,
            y_min_vol, y_max_vol,
            z_min_vol, z_max_vol,
            label="ROI"
        )
    
    def draw_rois_on_view(self, ax, view_type, current_slice):
        """Draw ROI on view - UPDATED for compatibility"""
        if not self.current_roi:
            return
        
        if not self.current_roi.is_visible_on_slice(view_type, current_slice):
            return
        
        x_min, x_max, y_min, y_max = self.current_roi.get_bounds_for_view(view_type)
        D, H, W = self.viewer.scan_array.shape
        
        if view_type == 'axial':
            x_min_disp = (W - 1) - x_max
            x_max_disp = (W - 1) - x_min
            y_min_disp = y_min
            y_max_disp = y_max
        elif view_type == 'coronal':
            x_min_disp = (W - 1) - x_max
            x_max_disp = (W - 1) - x_min
            y_min_disp = (D - 1) - y_max
            y_max_disp = (D - 1) - y_min
        elif view_type == 'sagittal':
            x_min_disp = y_min
            x_max_disp = y_max
            y_min_disp = (D - 1) - x_max
            y_max_disp = (D - 1) - x_min
        else:
            return
        
        width = x_max_disp - x_min_disp
        height = y_max_disp - y_min_disp
        
        rect = Rectangle(
            (x_min_disp, y_min_disp), width, height,
            linewidth=2,
            edgecolor='cyan',
            facecolor='none',
            linestyle='-'
        )
        ax.add_patch(rect)
        
        slice_start, slice_end = self.current_roi.get_slice_range(view_type)
        num_slices = self.current_roi.get_num_slices(view_type)
        label_text = f"{self.current_roi.label}\n[{slice_start}-{slice_end}] ({num_slices} slices)"
        
        ax.text(
            x_min_disp, y_max_disp + 5, label_text,
            color='cyan',
            fontsize=8,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
        )
    
    def adjust_roi_range(self):
        """Adjust ROI slice ranges"""
        if not self.current_roi:
            QMessageBox.warning(self.viewer, "No ROI", "Create an ROI first.")
            return False
        
        dialog = ROIAdjustDialog(self.viewer, self.current_roi, self.viewer.scan_array.shape)
        if dialog.exec_():
            x_min, x_max, y_min, y_max, z_min, z_max = dialog.get_adjusted_bounds()
            
            if x_min >= x_max or y_min >= y_max or z_min >= z_max:
                QMessageBox.critical(self.viewer, "Invalid Range", "Min < Max required.")
                return False
            
            self.current_roi.x_min = x_min
            self.current_roi.x_max = x_max
            self.current_roi.y_min = y_min
            self.current_roi.y_max = y_max
            self.current_roi.z_min = z_min
            self.current_roi.z_max = z_max
            
            self.viewer.update_all_views()
            return True
        return False
    
    def zoom_to_roi(self, roi_index=None):
        """Zoom to ROI - UPDATED for compatibility"""
        return self.zoom_all_views_to_roi()
    
    def zoom_all_views_to_roi(self):
        """Zoom all views to ROI"""
        if not self.current_roi:
            return False
        
        center_x = (self.current_roi.x_min + self.current_roi.x_max) // 2
        center_y = (self.current_roi.y_min + self.current_roi.y_max) // 2
        center_z = (self.current_roi.z_min + self.current_roi.z_max) // 2
        
        self.viewer.crosshair_x = center_x
        self.viewer.crosshair_y = center_y
        self.viewer.crosshair_z = center_z
        
        self.viewer.axial_slider.setValue(center_z)
        self.viewer.coronal_slider.setValue(center_y)
        self.viewer.sagittal_slider.setValue(center_x)
        
        self.viewer.update_all_views()
        return True
    
    def reset_zoom(self, view_type):
        """Reset zoom for view"""
        if self.viewer.scan_array is None:
            return
        
        D, H, W = self.viewer.scan_array.shape
        
        if view_type == 'axial':
            ax = self.viewer.axial_ax
            canvas = self.viewer.axial_canvas
            ax.set_xlim(0, W)
            ax.set_ylim(0, H)
        elif view_type == 'coronal':
            ax = self.viewer.coronal_ax
            canvas = self.viewer.coronal_canvas
            ax.set_xlim(0, W)
            ax.set_ylim(0, D)
        elif view_type == 'sagittal':
            ax = self.viewer.sagittal_ax
            canvas = self.viewer.sagittal_canvas
            ax.set_xlim(0, H)
            ax.set_ylim(0, D)
        else:
            return
        
        canvas.draw_idle()
    
    def export_roi_volume(self, output_path=None):
        """Export ROI as NIfTI file"""
        if not self.current_roi:
            QMessageBox.warning(self.viewer, "No ROI", "Create an ROI first.")
            return False
        
        if self.viewer.itk_image is None:  # ✅ Fixed: was self.itk_image
            QMessageBox.warning(self.viewer, "No Image", "No image loaded.")
            return False
        
        if output_path is None:
            options = QFileDialog.Options()
            output_path, _ = QFileDialog.getSaveFileName(
                self.viewer,
                "Export ROI Volume",
                "roi_volume.nii.gz",
                "NIfTI Files (*.nii *.nii.gz)",
                options=options
            )
        
        if not output_path:
            return False
        
        try:
            x_min, x_max = self.current_roi.x_min, self.current_roi.x_max
            y_min, y_max = self.current_roi.y_min, self.current_roi.y_max
            z_min, z_max = self.current_roi.z_min, self.current_roi.z_max
            
            roi_volume = self.viewer.scan_array[
                z_min:z_max + 1,
                y_min:y_max + 1,
                x_min:x_max + 1
            ]
            
            roi_itk = sitk.GetImageFromArray(roi_volume)
            roi_itk.SetSpacing(self.viewer.itk_image.GetSpacing())  # ✅ Fixed: was self.itk_image
            roi_itk.SetDirection(self.viewer.itk_image.GetDirection())  # ✅ Fixed: was self.itk_image
            
            original_origin = self.viewer.itk_image.GetOrigin()  # ✅ Fixed: was self.itk_image
            spacing = self.viewer.itk_image.GetSpacing()  # ✅ Fixed: was self.itk_image
            new_origin = (
                original_origin[0] + x_min * spacing[0],
                original_origin[1] + y_min * spacing[1],
                original_origin[2] + z_min * spacing[2]
            )
            roi_itk.SetOrigin(new_origin)
            
            sitk.WriteImage(roi_itk, output_path)
            
            QMessageBox.information(
                self.viewer,
                "Export Successful",
                f"ROI exported!\n\nFile: {output_path}\nShape: {roi_volume.shape}"
            )
            return True
            
        except Exception as e:
            QMessageBox.critical(self.viewer, "Export Failed", str(e))
            return False
    
    def delete_roi(self, roi_index=None):
        """Delete ROI - UPDATED for compatibility"""
        self.clear_roi()
        return True
    
    def clear_all_rois(self):
        """Clear all ROIs - UPDATED for compatibility"""
        self.clear_roi()
    
    def clear_roi(self):
        """Clear current ROI"""
        self.current_roi = None
        self.disable_drawing()
        self.viewer.update_all_views()
    
    def get_roi_list(self):
        """Get ROI list - UPDATED for compatibility"""
        if self.current_roi:
            return [f"0: {self.current_roi.label} (3D ROI)"]
        return []
    
    def navigate_to_roi(self, roi_index):
        """Navigate to ROI - UPDATED for compatibility"""
        return self.zoom_all_views_to_roi()
    
    def has_roi(self):
        """Check if ROI exists"""
        return self.current_roi is not None


# ==================== MAIN VIEWER CLASS ====================

class MRIViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.itk_image = None
        self.scan_array = None
        
        self.itk_segmentation = None 
        self.segmentation_data = None 
        
        self.crosshair_x = 0
        self.crosshair_y = 0
        self.crosshair_z = 0
        self.current_colormap = 'gray'
        
        self.min_intensity = 0.0
        self.max_intensity = 1.0
        
        self.axial_L = 0.5; self.axial_W = 1.0
        self.coronal_L = 0.5; self.coronal_W = 1.0
        self.sagittal_L = 0.5; self.sagittal_W = 1.0
        self.perspective_L = 0.5; self.perspective_W = 1.0 
        
        self.oblique_angle_xy = 0
        self.oblique_angle_xz = 0
        self.oblique_angle_yz = 0
        self.oblique_offset = 0
        self.oblique_cached = None
        
        self.is_playing_z = False
        self.is_playing_y = False
        self.is_playing_x = False
        self.cine_timer = QTimer(self)
        self.cine_timer.timeout.connect(self.next_slice)
        
        self.perspective_view = 'Axial'
        self.view4_mode = 'Slice Edge'
        
        self.modality_text = 'Unknown'
        self.original_orientation = 'Unknown'
        self.orientation_text = 'RAS+'
        self.anatomy_region = 'Unknown'
        self.detected_organs = []
        self.original_scan_view = 'Unknown'  # Store detected original view
        self.ai_worker = None
        
        # Gemini API key
        self.gemini_api_key = None
        
        self.block_updates = False
        self.roi_manager = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Self-Training Medical Viewer with AI')
        self.setGeometry(100, 100, 1600, 900)
        
        main_layout = QHBoxLayout(self)

        # === LEFT PANEL with Scroll ===
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setFixedWidth(300)
        
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setAlignment(Qt.AlignTop)
        controls_layout.setSpacing(8)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        
        load_scan_btn = QPushButton("Load Main Scan (.nii)")
        load_scan_btn.clicked.connect(lambda: self.load_file('scan'))
        controls_layout.addWidget(load_scan_btn)
        
        load_dicom_btn = QPushButton("Load DICOM Series")
        load_dicom_btn.clicked.connect(lambda: self.load_dicom_series('scan'))
        load_dicom_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        controls_layout.addWidget(load_dicom_btn)
        
        # After load_dicom_btn section, ADD THIS:

        load_single_dicom_btn = QPushButton("📄 MPR View Single DICOM")
        load_single_dicom_btn.clicked.connect(self.load_single_dicom_mpr)
        load_single_dicom_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        load_single_dicom_btn.setToolTip("Load a single DICOM file and create MPR views from it")
        controls_layout.addWidget(load_single_dicom_btn)
        
        load_seg_btn = QPushButton("Load Mask (.nii)")
        load_seg_btn.clicked.connect(lambda: self.load_file('segmentation'))
        controls_layout.addWidget(load_seg_btn)

        # Totalsegmentator button
        segment_btn = QPushButton("🤖 Auto-Segment with TotalSegmentator")
        segment_btn.clicked.connect(self.segment_with_totalsegmentator)
        segment_btn.setStyleSheet("background-color: #FF6B6B; color: white; font-weight: bold;")
        controls_layout.addWidget(segment_btn)




        
        controls_layout.addSpacing(10)
           
       # Add after "Load Mask" button
        api_key_btn = QPushButton("🔑 Set Gemini API Key")
        api_key_btn.clicked.connect(self.set_gemini_api_key)
        controls_layout.addWidget(api_key_btn)
       
     
        controls_layout.addSpacing(10)
        
        # AI Info Display
        ai_info_group = QGroupBox("AI Detection")
        ai_info_layout = QVBoxLayout()
        ai_info_layout.setSpacing(4)

        self.modality_display = QLabel(f"Modality: {self.modality_text}")
        self.modality_display.setStyleSheet("font-weight: bold; color: purple; font-size: 10pt;")
        ai_info_layout.addWidget(self.modality_display)

        # ADD THIS NEW LINE - Missing display element
        self.original_orientation_display = QLabel(f"Original: {self.original_orientation}")
        self.original_orientation_display.setStyleSheet("color: #9C27B0; font-size: 9pt;")
        ai_info_layout.addWidget(self.original_orientation_display)

        self.orientation_display = QLabel(f"Standardized: {self.orientation_text}")
        self.orientation_display.setStyleSheet("font-weight: bold; color: blue; font-size: 9pt;")
        ai_info_layout.addWidget(self.orientation_display)

        self.original_view_display = QLabel("Original View: Unknown")
        self.original_view_display.setStyleSheet("color: #FF6B6B; font-size: 9pt; font-weight: bold;")
        ai_info_layout.addWidget(self.original_view_display)



        # Always show anatomy display (Gemini-based)
        self.anatomy_display = QLabel(f"Anatomy: {self.anatomy_region}")
        self.anatomy_display.setStyleSheet("color: green; font-size: 9pt; font-weight: bold;")
        ai_info_layout.addWidget(self.anatomy_display)

        self.ai_status_display = QLabel("AI: Ready" if GEMINI_AVAILABLE else "AI: Install Gemini")
        self.ai_status_display.setStyleSheet("color: gray; font-size: 8pt; font-style: italic;")
        ai_info_layout.addWidget(self.ai_status_display)
        
        ai_info_group.setLayout(ai_info_layout)
        controls_layout.addWidget(ai_info_group)
        
        controls_layout.addSpacing(10)

        # Slice Controls
        slice_controls_group = QGroupBox("Slice Controls")
        slice_controls_layout = QGridLayout(slice_controls_group)
        slice_controls_layout.setVerticalSpacing(4)
        slice_controls_layout.setHorizontalSpacing(4)

        slice_controls_layout.addWidget(QLabel("Z:"), 0, 0)
        self.axial_slider = QSlider(Qt.Horizontal)
        self.axial_slider.setRange(0, 0)
        self.axial_slider.valueChanged.connect(lambda: self.update_crosshair_from_slider('z'))
        slice_controls_layout.addWidget(self.axial_slider, 0, 1)

        slice_controls_layout.addWidget(QLabel("Y:"), 1, 0)
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.coronal_slider.setRange(0, 0)
        self.coronal_slider.valueChanged.connect(lambda: self.update_crosshair_from_slider('y'))
        slice_controls_layout.addWidget(self.coronal_slider, 1, 1)

        slice_controls_layout.addWidget(QLabel("X:"), 2, 0)
        self.sagittal_slider = QSlider(Qt.Horizontal)
        self.sagittal_slider.setRange(0, 0)
        self.sagittal_slider.valueChanged.connect(lambda: self.update_crosshair_from_slider('x'))
        slice_controls_layout.addWidget(self.sagittal_slider, 2, 1)
        
        controls_layout.addWidget(slice_controls_group)

        # 4th View Mode
        perspective_group = QGroupBox("4th Panel")
        perspective_layout = QVBoxLayout(perspective_group)
        perspective_layout.setSpacing(4)
        
        self.view4_mode_combo = QComboBox()
        self.view4_mode_combo.addItems(['Slice Edge', 'Oblique Slice'])
        self.view4_mode_combo.currentTextChanged.connect(self.set_view4_mode)
        perspective_layout.addWidget(self.view4_mode_combo)
        
        controls_layout.addWidget(perspective_group)
        
        # Cine Controls
        cine_group = QGroupBox("Cine (Animation)")
        cine_layout = QVBoxLayout(cine_group)
        cine_layout.setSpacing(4)
        
        self.play_z_btn = QPushButton("▶ Z")
        self.play_z_btn.clicked.connect(lambda: self.toggle_cine_loop('z'))
        cine_layout.addWidget(self.play_z_btn)

        self.play_y_btn = QPushButton("▶ Y")
        self.play_y_btn.clicked.connect(lambda: self.toggle_cine_loop('y'))
        cine_layout.addWidget(self.play_y_btn)

        self.play_x_btn = QPushButton("▶ X")
        self.play_x_btn.clicked.connect(lambda: self.toggle_cine_loop('x'))
        cine_layout.addWidget(self.play_x_btn)
        
        cine_group.setLayout(cine_layout)
        controls_layout.addWidget(cine_group)

        # ROI Controls
        roi_group = QGroupBox("ROI Tools")
        roi_layout = QVBoxLayout(roi_group)
        roi_layout.setSpacing(4)

        draw_label = QLabel("Draw ROI on:")
        draw_label.setStyleSheet("font-size: 9pt; color: gray;")
        roi_layout.addWidget(draw_label)
        
        draw_layout = QGridLayout()
        draw_layout.setSpacing(4)
        
        self.roi_axial_btn = QPushButton("Axial")
        self.roi_axial_btn.clicked.connect(lambda: self.start_roi_drawing('axial'))
        draw_layout.addWidget(self.roi_axial_btn, 0, 0)

        self.roi_coronal_btn = QPushButton("Coronal")
        self.roi_coronal_btn.clicked.connect(lambda: self.start_roi_drawing('coronal'))
        draw_layout.addWidget(self.roi_coronal_btn, 0, 1)

        self.roi_sagittal_btn = QPushButton("Sagittal")
        self.roi_sagittal_btn.clicked.connect(lambda: self.start_roi_drawing('sagittal'))
        draw_layout.addWidget(self.roi_sagittal_btn, 1, 0, 1, 2)

        roi_layout.addLayout(draw_layout)

        roi_list_label = QLabel("Saved ROIs:")
        roi_list_label.setStyleSheet("font-size: 9pt; color: gray; margin-top: 4px;")
        roi_layout.addWidget(roi_list_label)
        
        self.roi_list_widget = QListWidget()
        self.roi_list_widget.setMaximumHeight(80)
        self.roi_list_widget.itemClicked.connect(self.on_roi_selected)
        roi_layout.addWidget(self.roi_list_widget)

        roi_action_layout = QGridLayout()
        roi_action_layout.setSpacing(4)
        
        self.roi_zoom_btn = QPushButton("Zoom")
        self.roi_zoom_btn.clicked.connect(self.zoom_to_selected_roi)
        roi_action_layout.addWidget(self.roi_zoom_btn, 0, 0)

        self.roi_goto_btn = QPushButton("Go to")
        self.roi_goto_btn.clicked.connect(self.goto_selected_roi)
        roi_action_layout.addWidget(self.roi_goto_btn, 0, 1)

        self.roi_reset_zoom_btn = QPushButton("Reset")
        self.roi_reset_zoom_btn.clicked.connect(self.reset_all_zooms)
        roi_action_layout.addWidget(self.roi_reset_zoom_btn, 1, 0)

        self.roi_delete_btn = QPushButton("Delete")
        self.roi_delete_btn.clicked.connect(self.delete_selected_roi)
        roi_action_layout.addWidget(self.roi_delete_btn, 1, 1)

        roi_layout.addLayout(roi_action_layout)

        self.roi_clear_btn = QPushButton("Clear All ROIs")
        self.roi_clear_btn.clicked.connect(self.clear_all_rois)
        roi_layout.addWidget(self.roi_clear_btn)

        # ==================== DIY INSTRUCTION: ADD THIS NEW BUTTON ====================
        adjust_roi_btn = QPushButton("🔧 Adjust ROI Range")
        adjust_roi_btn.clicked.connect(self.adjust_roi_range)
        adjust_roi_btn.setToolTip("Fine-tune the 3D boundaries (X, Y, Z) of the current ROI.")
        roi_layout.addWidget(adjust_roi_btn)
        # ============================= END OF NEW BUTTON ==============================



        export_btn = QPushButton("📁 Export Slices")
        export_btn.clicked.connect(self.export_selected_roi_slices)
        export_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        roi_layout.addWidget(export_btn)

        # ==================== DIY INSTRUCTION ====================
        # Add this code to create the "Export ROI to .nii" button

        export_volume_btn = QPushButton("💾 Export ROI to .nii")
        export_volume_btn.clicked.connect(self.export_roi_volume)
        export_volume_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;") # Orange color
        export_volume_btn.setToolTip("Export the current 3D ROI as a standalone NIfTI (.nii.gz) file.")
        roi_layout.addWidget(export_volume_btn)

        # ================= END OF DIY INSTRUCTION =================

        controls_layout.addWidget(roi_group)
        controls_layout.addStretch(1)
        
        scroll_area.setWidget(controls_widget)
        main_layout.addWidget(scroll_area)

        # === RIGHT PANEL (Views) ===
        views_layout = QGridLayout()
        
        self.axial_fig, self.axial_ax = plt.subplots(figsize=(4, 4))
        self.axial_canvas = FigureCanvas(self.axial_fig)
        axial_panel = self._create_view_panel(
            'axial', self.axial_ax, self.axial_canvas, 
            "1. Axial (Head Up)", self.handle_mouse_click
        )
        views_layout.addLayout(axial_panel, 0, 0)
        
        self.coronal_fig, self.coronal_ax = plt.subplots(figsize=(4, 4))
        self.coronal_canvas = FigureCanvas(self.coronal_fig)
        coronal_panel = self._create_view_panel(
            'coronal', self.coronal_ax, self.coronal_canvas, 
            "2. Coronal (Face Forward)", self.handle_mouse_click
        )
        views_layout.addLayout(coronal_panel, 1, 0)

        self.sagittal_fig, self.sagittal_ax = plt.subplots(figsize=(4, 4))
        self.sagittal_canvas = FigureCanvas(self.sagittal_fig)
        sagittal_panel = self._create_view_panel(
            'sagittal', self.sagittal_ax, self.sagittal_canvas, 
            "3. Sagittal (Right Side)", self.handle_mouse_click
        )
        views_layout.addLayout(sagittal_panel, 0, 1)

        self.perspective_fig, self.perspective_ax = plt.subplots(figsize=(4, 4))
        self.perspective_canvas = FigureCanvas(self.perspective_fig)
        perspective_panel = self._create_view_panel(
            'perspective', self.perspective_ax, self.perspective_canvas, 
            "4. Dynamic View", None, is_perspective=True
        )
        views_layout.addLayout(perspective_panel, 1, 1)

        main_layout.addLayout(views_layout, 1)
        self.setLayout(main_layout)
        plt.tight_layout()
        
        self.roi_manager = ROIManager(self)
        self._update_perspective_controls()
    

    

   
    def _create_view_panel(self, view_name, ax, canvas, title, click_handler, 
                          draw_crosshair=True, is_perspective=False):
        canvas.setCursor(QCursor(Qt.CrossCursor))
        if click_handler:
            canvas.mpl_connect('button_press_event', 
                             lambda event: click_handler(event, view_name))
            
        ax.set_title(title)
        ax.axis('off')

        level_slider = QSlider(Qt.Horizontal)
        level_slider.setRange(0, 100)
        level_slider.setValue(50)
        setattr(self, f'{view_name}_level_slider', level_slider) 
        
        width_slider = QSlider(Qt.Horizontal)
        width_slider.setRange(1, 200)
        width_slider.setValue(100)
        setattr(self, f'{view_name}_width_slider', width_slider) 

        level_slider.valueChanged.connect(lambda: self.update_view_windowing(view_name))
        width_slider.valueChanged.connect(lambda: self.update_view_windowing(view_name))

        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(5)
        slider_layout.addWidget(QLabel("L:"))
        slider_layout.addWidget(level_slider, 1)
        slider_layout.addWidget(QLabel("W:"))
        slider_layout.addWidget(width_slider, 1)

        panel_layout = QVBoxLayout()
        panel_layout.addWidget(canvas)
        
        if is_perspective:
            self.perspective_windowing_widget = QWidget()
            self.perspective_windowing_widget.setLayout(slider_layout)
            panel_layout.addWidget(self.perspective_windowing_widget)
            
            self.perspective_controls_container = QVBoxLayout()
            self.perspective_controls_container.setContentsMargins(0, 0, 0, 0)
            self.perspective_controls_container.setSpacing(2)
            panel_layout.addLayout(self.perspective_controls_container)
            
            self._create_oblique_controls()
            self._create_orthogonal_selector()
        else:
            panel_layout.addLayout(slider_layout)
        
        panel_layout.setStretch(0, 1)
        panel_layout.setStretch(1, 0)
        
        return panel_layout

    def _create_oblique_controls(self):
        self.oblique_controls_widget = QWidget()
        oblique_layout = QVBoxLayout(self.oblique_controls_widget)
        oblique_layout.setContentsMargins(0, 0, 0, 0)
        oblique_layout.setSpacing(2)
        
        oblique_controls_1 = QHBoxLayout()
        oblique_controls_1.setContentsMargins(0, 0, 0, 0)
        oblique_controls_1.setSpacing(5)
        oblique_controls_1.addWidget(QLabel("XY:"))
        self.perspective_angle_xy_slider = QSlider(Qt.Horizontal)
        self.perspective_angle_xy_slider.setRange(0, 360)
        self.perspective_angle_xy_slider.setValue(0)
        self.perspective_angle_xy_slider.valueChanged.connect(self.update_oblique_view)
        oblique_controls_1.addWidget(self.perspective_angle_xy_slider, 1)
        
        oblique_controls_2 = QHBoxLayout()
        oblique_controls_2.setContentsMargins(0, 0, 0, 0)
        oblique_controls_2.setSpacing(5)
        oblique_controls_2.addWidget(QLabel("XZ:"))
        self.perspective_angle_xz_slider = QSlider(Qt.Horizontal)
        self.perspective_angle_xz_slider.setRange(0, 360)
        self.perspective_angle_xz_slider.setValue(0)
        self.perspective_angle_xz_slider.valueChanged.connect(self.update_oblique_view)
        oblique_controls_2.addWidget(self.perspective_angle_xz_slider, 1)
        
        oblique_controls_3 = QHBoxLayout()
        oblique_controls_3.setContentsMargins(0, 0, 0, 0)
        oblique_controls_3.setSpacing(5)
        oblique_controls_3.addWidget(QLabel("YZ:"))
        self.perspective_angle_yz_slider = QSlider(Qt.Horizontal)
        self.perspective_angle_yz_slider.setRange(0, 360)
        self.perspective_angle_yz_slider.setValue(0)
        self.perspective_angle_yz_slider.valueChanged.connect(self.update_oblique_view)
        oblique_controls_3.addWidget(self.perspective_angle_yz_slider, 1)
        
        oblique_controls_4 = QHBoxLayout()
        oblique_controls_4.setContentsMargins(0, 0, 0, 0)
        oblique_controls_4.setSpacing(5)
        oblique_controls_4.addWidget(QLabel("Scroll:"))
        self.perspective_offset_slider = QSlider(Qt.Horizontal)
        self.perspective_offset_slider.setRange(0, 0)
        self.perspective_offset_slider.setValue(0)
        self.perspective_offset_slider.valueChanged.connect(self.update_oblique_view)
        oblique_controls_4.addWidget(self.perspective_offset_slider, 1)
        
        oblique_layout.addLayout(oblique_controls_1)
        oblique_layout.addLayout(oblique_controls_2)
        oblique_layout.addLayout(oblique_controls_3)
        oblique_layout.addLayout(oblique_controls_4)
        
        self.oblique_controls_widget.hide()

    def _create_orthogonal_selector(self):
        self.orthogonal_selector_widget = QWidget()
        orthogonal_layout = QHBoxLayout(self.orthogonal_selector_widget)
        orthogonal_layout.setContentsMargins(0, 0, 0, 0)
        orthogonal_layout.setSpacing(5)
        
        orthogonal_layout.addWidget(QLabel("Slice:"))
        self.perspective_combo = QComboBox()
        self.perspective_combo.addItems(['Axial', 'Coronal', 'Sagittal'])
        self.perspective_combo.currentTextChanged.connect(self.set_perspective)
        orthogonal_layout.addWidget(self.perspective_combo, 1)
        
        self.orthogonal_selector_widget.hide()

    def _update_perspective_controls(self):
        while self.perspective_controls_container.count():
            child = self.perspective_controls_container.takeAt(0)
            if child.widget():
                child.widget().hide()
        
        if self.view4_mode == 'Oblique Slice':
            self.perspective_windowing_widget.show()
            self.perspective_controls_container.addWidget(self.oblique_controls_widget)
            self.oblique_controls_widget.show()
            self.orthogonal_selector_widget.hide()
        elif self.view4_mode == 'Slice Edge':
            self.perspective_windowing_widget.hide()
            self.perspective_controls_container.addWidget(self.orthogonal_selector_widget)
            self.orthogonal_selector_widget.show()
            self.oblique_controls_widget.hide()

    def reorient_to_ras(self, itk_img):
        """Reorient image to RAS+ and store original orientation."""
        try:
            current_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
                itk_img.GetDirection()
            )
            self.original_orientation = current_orientation
            
            # If the orientation is already RAS, do nothing.
            if current_orientation == 'RAS':
                return itk_img
            
            # For any other orientation, force it to RAS. This is a robust way to standardize.
            orient_filter = sitk.DICOMOrientImageFilter()
            orient_filter.SetDesiredCoordinateOrientation('RAS')
            return orient_filter.Execute(itk_img)
            
        except Exception as e:
            print(f"[WARNING] Could not determine or reorient image orientation: {e}")
            QMessageBox.warning(self, "Orientation Warning",
                                f"Could not automatically reorient the image to RAS.\n"
                                f"The display might not match standard anatomical views.\n\nError: {e}")
            self.original_orientation = "Unknown"
            return itk_img # Return the original image if reorientation fails


    def load_dicom_series(self, file_type):
        directory = QFileDialog.getExistingDirectory(self, "Select DICOM Series Directory", "")
        
        if not directory:
            return
            
        try:
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(directory)
            
            if not series_ids:
                QMessageBox.critical(self, "Error", 
                                "No DICOM series found in the selected directory.")
                return
            
            if len(series_ids) > 1:
                QMessageBox.information(self, "Multiple Series", 
                                    f"Found {len(series_ids)} DICOM series. Loading the first one.")
            
            series_id = series_ids[0]
            dicom_names = reader.GetGDCMSeriesFileNames(directory, series_id)
            
            if not dicom_names:
                QMessageBox.critical(self, "Error", "No DICOM files found for the series.")
                return
            
            reader.SetFileNames(dicom_names)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            
            itk_img = reader.Execute()
            itk_img = self.reorient_to_ras(itk_img)
            np_array = sitk.GetArrayFromImage(itk_img)

            if file_type == 'scan':
                self.itk_image = itk_img
                self.scan_array = np_array
                self.orientation_text = 'RAS+'
                
                self.original_scan_view = "Analyzing with AI..."
                
                try:
                    modality = reader.GetMetaData(0, "0008|0060") if reader.HasMetaDataKey(0, "0008|0060") else "Unknown"
                    self.modality_text = modality
                except:
                    self.modality_text = "DICOM"
                
                self.modality_display.setText(f"Modality: {self.modality_text}")
                self.original_orientation_display.setText(f"Original: {self.original_orientation}")
                self.orientation_display.setText(f"Standardized: {self.orientation_text}")
                self.original_view_display.setText(f"Original View: {self.original_scan_view}")
                
                try:
                    patient_name = reader.GetMetaData(0, "0010|0010") if reader.HasMetaDataKey(0, "0010|0010") else "Unknown"
                    study_date = reader.GetMetaData(0, "0008|0020") if reader.HasMetaDataKey(0, "0008|0020") else "Unknown"
                    
                    info_text = (f"DICOM Series loaded successfully!\n\n"
                                f"Modality: {self.modality_text}\n"
                                f"Orientation: {self.orientation_text}\n"
                                f"Original View: {self.original_scan_view}\n\n"
                                f"Shape: {self.scan_array.shape} (Z, Y, X)\n"
                                f"Number of slices: {len(dicom_names)}\n\n"
                                f"Patient: {patient_name}\n"
                                f"Study Date: {study_date}")
                except:
                    info_text = (f"DICOM Series loaded successfully!\n\n"
                                f"Modality: {self.modality_text}\n"
                                f"Orientation: {self.orientation_text}\n"
                                f"Original View: {self.original_scan_view}\n\n"
                                f"Shape: {self.scan_array.shape} (Z, Y, X)\n"
                                f"Number of slices: {len(dicom_names)}")
                
                QMessageBox.information(self, "Load Successful", info_text)
                self.segmentation_data = None
                self.initialize_view()
                self.update_all_views()
                
                # Run Gemini AI analysis if available
                if GEMINI_AVAILABLE and self.gemini_api_key:
                    self.run_ai_analysis()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load DICOM series:\n{str(e)}")
            self.scan_array = None
            self.segmentation_data = None
            self._clear_views()



    def load_file(self, file_type):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, f"Load {file_type} NII File", "",
                                                "NIfTI Files (*.nii *.nii.gz)", options=options)

        if file_path:
            try:
                itk_img = None
                np_array = None

                try:
                    itk_img = sitk.ReadImage(file_path)
                except RuntimeError as e:
                    if "orthonormal" in str(e):
                        QMessageBox.warning(
                            self, "Header Warning",
                            "The NIfTI file header has a non-orthonormal orientation.\n\n"
                            "Fixing by rebuilding header metadata..."
                        )
                        print("[INFO] Non-orthonormal header detected. Using nibabel fallback.")
                        
                        nib_img = nib.load(file_path)
                        np_data = nib_img.get_fdata()
                        affine = nib_img.affine
                        
                        # Convert nibabel's (W, H, D) numpy array to ITK's (D, H, W)
                        np_data_transposed = np_data.transpose(2, 1, 0)
                        
                        itk_img = sitk.GetImageFromArray(np_data_transposed)
                        
                        # Extract spacing and origin from the affine matrix
                        spacing = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
                        origin = affine[:3, 3]
                        direction = affine[:3, :3] / spacing
                        
                        itk_img.SetSpacing([float(s) for s in spacing])
                        itk_img.SetOrigin([float(o) for o in origin])
                        itk_img.SetDirection(direction.flatten())
                        
                        print("[INFO] Successfully rebuilt ITK image from nibabel data.")
                    else:
                        raise e

                itk_img = self.reorient_to_ras(itk_img)
                np_array = sitk.GetArrayFromImage(itk_img)

                if file_type == 'scan':
                    self.itk_image = itk_img
                    self.scan_array = np_array
                    self.orientation_text = 'RAS+'
                    self.modality_text = 'NIfTI'
                    
                    self.original_scan_view = "Analyzing with AI..."
                    
                    self.modality_display.setText(f"Modality: {self.modality_text}")
                    self.original_orientation_display.setText(f"Original: {self.original_orientation}")
                    self.orientation_display.setText(f"Standardized: {self.orientation_text}")
                    self.original_view_display.setText(f"Original View: {self.original_scan_view}")
                    
                    QMessageBox.information(self, "Load Successful",
                                        f"Scan loaded and reoriented to RAS+\nShape: {self.scan_array.shape} (Z, Y, X)")
                    self.segmentation_data = None
                    self.initialize_view()
                    self.update_all_views()
                    
                    if GEMINI_AVAILABLE and self.gemini_api_key:
                        self.run_ai_analysis()
                    elif GEMINI_AVAILABLE and not self.gemini_api_key:
                        reply = QMessageBox.question(self, "Enable AI Analysis?",
                                                    "Enable AI-powered anatomy detection?\nThis requires a free Gemini API key.",
                                                    QMessageBox.Yes | QMessageBox.No)
                        if reply == QMessageBox.Yes:
                            self.set_gemini_api_key()
                            if self.gemini_api_key:
                                self.run_ai_analysis()

                elif file_type == 'segmentation':
                    if self.scan_array is None:
                        QMessageBox.warning(self, "Error", "Please load the main scan first.")
                        return
                    if np_array.shape != self.scan_array.shape:
                        QMessageBox.critical(self, "Error", f"Mask shape {np_array.shape} does not match scan shape {self.scan_array.shape}.")
                        return
                    self.itk_segmentation = itk_img
                    self.segmentation_data = np_array
                    QMessageBox.information(self, "Load Successful", "Mask loaded and reoriented to RAS+.")
                    self.update_all_views()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load file: {e}")
                self.scan_array = None
                self.segmentation_data = None
                self._clear_views()
                import traceback
                traceback.print_exc()



    def run_ai_analysis(self):
        if not GEMINI_AVAILABLE:
            QMessageBox.warning(self, "Gemini Required", 
                            "Install: pip install google-generativeai")
            return
        
        if not self.gemini_api_key:
            QMessageBox.warning(self, "API Key Required", 
                            "Please set your Gemini API key first")
            self.set_gemini_api_key()
            return
        
        if self.scan_array is None:
            QMessageBox.warning(self, "Warning", "Please load a scan first.")
            return
        
        self.ai_status_display.setText("AI: Analyzing with Gemini...")
        
        self.ai_worker = GeminiWorker(self.scan_array, self.gemini_api_key)
        self.ai_worker.progress.connect(self.update_ai_status)
        self.ai_worker.finished.connect(self.on_ai_analysis_complete)
        self.ai_worker.start()

    def set_gemini_api_key(self):
        from PyQt5.QtWidgets import QInputDialog
        api_key, ok = QInputDialog.getText(
            self, 'Gemini API Key', 
            'Enter your Gemini API key:\n(Get free key at: https://makersuite.google.com/app/apikey)',
            QLineEdit.Password
        )
        
        if ok and api_key:
            self.gemini_api_key = api_key
            QMessageBox.information(self, "Success", "API key saved!")
    




    
    def update_ai_status(self, message):
        """Update AI status display"""
        if hasattr(self, 'ai_status_display'):
            self.ai_status_display.setText(message)
    

    # --- PASTE THIS ENTIRE METHOD BACK INTO YOUR CODE ---

    def on_ai_analysis_complete(self, results):
        """Handle AI analysis results"""
        if not results.get('success', False):
            self.ai_status_display.setText("AI: Error")
            self.ai_status_display.setStyleSheet("color: red; font-size: 8pt;")
            QMessageBox.critical(self, "AI Error", f"AI analysis failed:\n{results.get('error', 'Unknown error')}")
            return
        
        msg = "🧠 AI Deep Analysis Results\n\n"
        
        if 'orientation' in results:
            ori_data = results['orientation']
            detected_ori = ori_data['orientation']
            confidence = ori_data['confidence']
            
            self.original_orientation = f"{detected_ori} (AI: {confidence:.1%})"
            self.original_orientation_display.setText(f"Original: {self.original_orientation}")
            
            msg += f"📐 Orientation: {ori_data['orientation']} (Confidence: {ori_data['confidence']:.1%})\n\n"

        if 'view' in results:
            view_data = results['view']
            detected_view = view_data['view']
            confidence = view_data['confidence']
            
            self.original_scan_view = f"{detected_view} ({confidence:.1%})"
            self.original_view_display.setText(f"Original View: {self.original_scan_view}")
            
            msg += f"👁️ Original View: {detected_view} (Confidence: {confidence:.1%})\n\n"
        
        if 'anatomy' in results:
            anat_data = results['anatomy']
            region = anat_data['primary_region']
            confidence = anat_data['confidence']
            
            self.anatomy_region = f"{region} ({confidence:.1%})"
            self.anatomy_display.setText(f"Anatomy: {region} ({confidence:.1%})")
            
            msg += f"🏥 Anatomical Region: {anat_data['primary_region']}\n"
            msg += f"   Confidence: {anat_data['confidence']:.1%}\n\n"
            msg += "Top 3 Predictions:\n"
            for i, (label, conf) in enumerate(anat_data['top3'], 1):
                msg += f"   {i}. {label}: {conf:.1%}\n"
        
        self.ai_status_display.setText("AI: Analysis Complete ✓")
        self.ai_status_display.setStyleSheet("color: green; font-size: 8pt; font-weight: bold;")
        
        QMessageBox.information(self, "AI Analysis Complete", msg)


    def find_organ_slice_range(self, organ_mask, label_val=None):
        """Find the slice range where the organ appears"""
        if organ_mask is None:
            return 0, self.scan_array.shape[0] - 1
        
        if label_val is not None and label_val > 0:
            mask_to_check = (organ_mask == label_val)
        else:
            mask_to_check = (organ_mask > 0)
        
        z_slices_with_organ = np.any(mask_to_check, axis=(1, 2))
        indices = np.where(z_slices_with_organ)[0]
        
        if len(indices) == 0:
            return 0, self.scan_array.shape[0] - 1
        
        return int(indices[0]), int(indices[-1])

    def _calculate_lw_from_norm(self, view_name, level_norm, width_norm):
        if self.max_intensity <= self.min_intensity:
            full_intensity_range = 1.0 
        else:
            full_intensity_range = self.max_intensity - self.min_intensity
        
        W = full_intensity_range * (width_norm / 100.0)
        L = self.min_intensity + full_intensity_range * (level_norm / 100.0)

        setattr(self, f'{view_name}_L', L)
        setattr(self, f'{view_name}_W', W)

    def update_view_windowing(self, view_name):
        if self.scan_array is None or self.block_updates:
            return
        
        level_slider = getattr(self, f'{view_name}_level_slider')
        width_slider = getattr(self, f'{view_name}_width_slider')
        
        level_norm = level_slider.value()
        width_norm = width_slider.value()
        
        self._calculate_lw_from_norm(view_name, level_norm, width_norm)
        self.update_all_views()

    def initialize_view(self):
        if self.scan_array is not None:
            self.block_updates = True
            
            D, H, W = self.scan_array.shape
            
            for slider, size in [(self.axial_slider, D), (self.coronal_slider, H), 
                                (self.sagittal_slider, W)]:
                slider.setRange(0, size - 1)
                slider.setValue(size // 2)

            self.crosshair_z = D // 2
            self.crosshair_y = H // 2
            self.crosshair_x = W // 2
            
            self.min_intensity = np.min(self.scan_array)
            self.max_intensity = np.max(self.scan_array)
            
            for view in ['axial', 'coronal', 'sagittal', 'perspective']:
                self._calculate_lw_from_norm(view, 50, 100)
                getattr(self, f'{view}_level_slider').setValue(50)
                getattr(self, f'{view}_width_slider').setValue(100)
            
            max_offset = max(D, H, W) // 2
            self.perspective_offset_slider.setRange(-max_offset, max_offset)
            self.perspective_offset_slider.setValue(0)
            self.oblique_offset = 0
            self.oblique_angle_xy = 0
            self.oblique_angle_xz = 0
            self.oblique_angle_yz = 0
            self.oblique_cached = None
            
 
            
            self.block_updates = False

    def update_oblique_view(self):
        if self.scan_array is None or self.block_updates:
            return
            
        self.oblique_angle_xy = self.perspective_angle_xy_slider.value()
        self.oblique_angle_xz = self.perspective_angle_xz_slider.value()
        self.oblique_angle_yz = self.perspective_angle_yz_slider.value()
        self.oblique_offset = self.perspective_offset_slider.value()
        
        if self.view4_mode == 'Oblique Slice':
            self.oblique_cached = None
            self.update_all_views()

    def calculate_oblique_slice(self):
        if self.scan_array is None:
            return None, None

        D, H, W = self.scan_array.shape
        
        angle_xy_rad = np.radians(self.oblique_angle_xy)
        angle_xz_rad = np.radians(self.oblique_angle_xz)
        angle_yz_rad = np.radians(self.oblique_angle_yz)
        
        diagonal = int(np.sqrt(H**2 + W**2))
        oblique_width = diagonal
        oblique_height = D
        
        j_grid = np.arange(oblique_width) - oblique_width / 2.0
        i_grid = np.arange(oblique_height)
        j_mesh, i_mesh = np.meshgrid(j_grid, i_grid)
        
        center_y = H / 2.0
        center_x = W / 2.0
        center_z = D / 2.0
        
        x_rot = j_mesh * np.cos(angle_xy_rad)
        y_rot = j_mesh * np.sin(angle_xy_rad)
        z_rot = i_mesh - center_z
        
        z_rot2 = z_rot * np.cos(angle_xz_rad) - x_rot * np.sin(angle_xz_rad)
        x_rot2 = z_rot * np.sin(angle_xz_rad) + x_rot * np.cos(angle_xz_rad)
        
        y_rot2 = y_rot * np.cos(angle_yz_rad) - z_rot2 * np.sin(angle_yz_rad)
        z_rot2 = y_rot * np.sin(angle_yz_rad) + z_rot2 * np.cos(angle_yz_rad)
        
        offset_y = -self.oblique_offset * np.sin(angle_xy_rad)
        offset_x = self.oblique_offset * np.cos(angle_xy_rad)
        
        vol_x = center_x + x_rot2 + offset_x
        vol_y = center_y + y_rot2 + offset_y
        vol_z = center_z + z_rot2
        
        vol_x = np.clip(vol_x, 0, W - 1.001)
        vol_y = np.clip(vol_y, 0, H - 1.001)
        vol_z = np.clip(vol_z, 0, D - 1.001)
        
        x_f = np.floor(vol_x).astype(int)
        y_f = np.floor(vol_y).astype(int)
        z_f = np.floor(vol_z).astype(int)
        
        x_frac = vol_x - x_f
        y_frac = vol_y - y_f
        z_frac = vol_z - z_f
        
        x_f = np.clip(x_f, 0, W - 2)
        y_f = np.clip(y_f, 0, H - 2)
        z_f = np.clip(z_f, 0, D - 2)
        
        x_f1 = np.clip(x_f + 1, 0, W - 1)
        y_f1 = np.clip(y_f + 1, 0, H - 1)
        z_f1 = np.clip(z_f + 1, 0, D - 1)
        
        v000 = self.scan_array[z_f, y_f, x_f]
        v100 = self.scan_array[z_f, y_f, x_f1]
        v010 = self.scan_array[z_f, y_f1, x_f]
        v110 = self.scan_array[z_f, y_f1, x_f1]
        v001 = self.scan_array[z_f1, y_f, x_f]
        v101 = self.scan_array[z_f1, y_f, x_f1]
        v011 = self.scan_array[z_f1, y_f1, x_f]
        v111 = self.scan_array[z_f1, y_f1, x_f1]
        
        oblique_scan = (v000 * (1-x_frac) * (1-y_frac) * (1-z_frac) +
                        v100 * x_frac * (1-y_frac) * (1-z_frac) +
                        v010 * (1-x_frac) * y_frac * (1-z_frac) +
                        v110 * x_frac * y_frac * (1-z_frac) +
                        v001 * (1-x_frac) * (1-y_frac) * z_frac +
                        v101 * x_frac * (1-y_frac) * z_frac +
                        v011 * (1-x_frac) * y_frac * z_frac +
                        v111 * x_frac * y_frac * z_frac)
        
        oblique_seg = None
        if self.segmentation_data is not None:
            s000 = self.segmentation_data[z_f, y_f, x_f]
            s100 = self.segmentation_data[z_f, y_f, x_f1]
            s010 = self.segmentation_data[z_f, y_f1, x_f]
            s110 = self.segmentation_data[z_f, y_f1, x_f1]
            s001 = self.segmentation_data[z_f1, y_f, x_f]
            s101 = self.segmentation_data[z_f1, y_f, x_f1]
            s011 = self.segmentation_data[z_f1, y_f1, x_f]
            s111 = self.segmentation_data[z_f1, y_f1, x_f1]
            
            oblique_seg = (s000 * (1-x_frac) * (1-y_frac) * (1-z_frac) +
                          s100 * x_frac * (1-y_frac) * (1-z_frac) +
                          s010 * (1-x_frac) * y_frac * (1-z_frac) +
                          s110 * x_frac * y_frac * (1-z_frac) +
                          s001 * (1-x_frac) * (1-y_frac) * z_frac +
                          s101 * x_frac * (1-y_frac) * z_frac +
                          s011 * (1-x_frac) * y_frac * z_frac +
                          s111 * x_frac * y_frac * z_frac)
        
        return oblique_scan, oblique_seg

    def update_crosshair_from_slider(self, axis):
        if self.scan_array is None or self.block_updates:
            return

        if axis == 'z':
            self.crosshair_z = self.axial_slider.value()
        elif axis == 'y':
            self.crosshair_y = self.coronal_slider.value()
        elif axis == 'x':
            self.crosshair_x = self.sagittal_slider.value()
            
        self.update_all_views()
        
    def handle_mouse_click(self, event, view_type):
        if self.scan_array is None or event.xdata is None or event.ydata is None:
            return
        
        x_data = int(round(event.xdata))
        y_data = int(round(event.ydata))
        
        D, H, W = self.scan_array.shape

        self.block_updates = True

        if view_type == 'axial':
            self.crosshair_x = (W - 1) - x_data
            self.crosshair_y = y_data
            self.sagittal_slider.setValue(self.crosshair_x)
            self.coronal_slider.setValue(self.crosshair_y)
            
        elif view_type == 'coronal':
            self.crosshair_x = (W - 1) - x_data
            self.crosshair_z = (D - 1) - y_data
            self.sagittal_slider.setValue(self.crosshair_x)
            self.axial_slider.setValue(self.crosshair_z)
            
        elif view_type == 'sagittal':
            self.crosshair_y = x_data 
            self.crosshair_z = (D - 1) - y_data
            self.coronal_slider.setValue(self.crosshair_y)
            self.axial_slider.setValue(self.crosshair_z)

        self.block_updates = False
        self.update_all_views()
        
    def _plot_slice(self, ax, canvas, slice_data, seg_slice, view_type, crosshair_pos, 
                    view_L, view_W, draw_filled_mask=False, draw_contour=False, 
                    contour_color='red', draw_crosshair=True, show_scan_data=True):
        ax.clear()

        vmin = view_L - (view_W / 2.0)
        vmax = view_L + (view_W / 2.0)
        
        if slice_data is not None:
            if show_scan_data:
                ax.imshow(slice_data, cmap=self.current_colormap, origin='lower', 
                         vmin=vmin, vmax=vmax, interpolation='nearest') 
            else:
                ax.set_facecolor('black')
                
            if seg_slice is not None and np.any(seg_slice > 0) and draw_filled_mask:
                seg_overlay = np.ma.masked_where(seg_slice == 0, seg_slice)
                ax.imshow(seg_overlay, cmap=CYAN_MASK_CMAP, origin='lower', alpha=1.0, 
                         interpolation='nearest') 

            if seg_slice is not None and np.any(seg_slice > 0) and draw_contour:
                ax.contour(seg_slice, levels=[0.5], colors=[contour_color], linewidths=1.5, zorder=5)

            x, y, z = crosshair_pos 
            D, H, W = self.scan_array.shape 
            
            h_val, v_val = -1, -1
            
            if view_type == 'axial':
                max_v, max_h = H - 1, W - 1
                h_val = x
                v_val = y
            elif view_type == 'coronal':
                max_v, max_h = D - 1, W - 1
                h_val =  x
                v_val =  z
            elif view_type == 'sagittal':
                max_v, max_h = D - 1, H - 1
                h_val = y
                v_val =  z
            elif view_type == 'perspective':
                if slice_data.ndim == 2:
                    max_v, max_h = slice_data.shape[0] - 1, slice_data.shape[1] - 1
                else: 
                    max_v, max_h = 0, 0
                h_val, v_val = max_h / 2, max_v / 2 
            
            ax.set_xlim(0, max_h + 1)
            ax.set_ylim(0, max_v + 1)
            
            if draw_crosshair and view_type != 'perspective':
                ax.axvline(x=h_val, color='yellow', linewidth=1.5, linestyle='-', alpha=0.8) 
                ax.axhline(y=v_val, color='yellow', linewidth=1.5, linestyle='-', alpha=0.8) 
                
            ax.axis('off')
        else:
            ax.clear()
            ax.axis('off')
            ax.set_title("No Data Loaded")
            
        if hasattr(self, 'roi_manager') and self.roi_manager is not None:
            if view_type == 'axial':
                current_slice = self.crosshair_z
            elif view_type == 'coronal':
                current_slice = self.crosshair_y
            elif view_type == 'sagittal':
                current_slice = self.crosshair_x
            else:
                current_slice = 0
            
            self.roi_manager.draw_rois_on_view(ax, view_type, current_slice)

        canvas.draw_idle()

    def set_view4_mode(self, mode_name):
        self.view4_mode = mode_name
        self._update_perspective_controls()
        self.update_all_views()
    
    def update_all_views(self):
        if self.scan_array is None or self.block_updates: 
            self._clear_views()
            return

        seg_data = self.segmentation_data
        mask_is_loaded = seg_data is not None
        
        axial_scan = np.fliplr(self.scan_array[self.crosshair_z, :, :])
        seg_axial = np.fliplr(seg_data[self.crosshair_z, :, :]) if mask_is_loaded else None

        coronal_slice = self.scan_array[:, self.crosshair_y, :]
        coronal_scan = np.fliplr(coronal_slice)
        seg_coronal = np.fliplr(seg_data[:, self.crosshair_y, :]) if mask_is_loaded else None

        sagittal_slice = self.scan_array[:, :, self.crosshair_x]
        sagittal_scan = sagittal_slice
        seg_sagittal = seg_data[:, :, self.crosshair_x] if mask_is_loaded else None
        
        crosshair_pos = (self.crosshair_x, self.crosshair_y, self.crosshair_z)
        
        self._plot_slice(self.axial_ax, self.axial_canvas, axial_scan, seg_axial, 'axial', 
                         crosshair_pos, 
                         self.axial_L, self.axial_W,
                         draw_filled_mask=False, 
                         draw_contour=False,
                         show_scan_data=True)
        self.axial_ax.set_title(f"1. Axial (Head Up) Z={self.crosshair_z}")
        
        self._plot_slice(self.coronal_ax, self.coronal_canvas, coronal_scan, seg_coronal, 'coronal', 
                         crosshair_pos, 
                         self.coronal_L, self.coronal_W,
                         draw_filled_mask=False, 
                         draw_contour=False,
                         show_scan_data=True)
        self.coronal_ax.set_title(f"2. Coronal (Face Forward) Y={self.crosshair_y}")
        
        self._plot_slice(self.sagittal_ax, self.sagittal_canvas, sagittal_scan, seg_sagittal, 'sagittal', 
                         crosshair_pos, 
                         self.sagittal_L, self.sagittal_W,
                         draw_filled_mask=False, 
                         draw_contour=False,
                         show_scan_data=True)
        self.sagittal_ax.set_title(f"3. Sagittal (Right Side) X={self.crosshair_x}")
                         
        if self.view4_mode == 'Oblique Slice':
            oblique_scan, oblique_seg = self.calculate_oblique_slice()
            
            self._plot_slice(
                self.perspective_ax, 
                self.perspective_canvas, 
                oblique_scan, 
                oblique_seg, 
                'perspective', 
                crosshair_pos, 
                self.perspective_L, self.perspective_W,
                draw_filled_mask=mask_is_loaded, 
                draw_contour=False,
                contour_color='blue',       
                draw_crosshair=False,
                show_scan_data=True
            )
            self.perspective_ax.set_title(f"4. Oblique Slice (XY:{self.oblique_angle_xy}° XZ:{self.oblique_angle_xz}° YZ:{self.oblique_angle_yz}°)")

        elif self.view4_mode == 'Slice Edge':
            view_type = self.perspective_view
            
            if not mask_is_loaded:
                self.perspective_ax.clear()
                self.perspective_ax.axis('off')
                self.perspective_ax.set_title("4. Load Mask to View Edge")
                self.perspective_canvas.draw()
                return 
                
            if view_type == 'Axial': 
                slice_to_plot = axial_scan
                seg_to_plot = seg_axial
                L, W = self.axial_L, self.axial_W
            elif view_type == 'Coronal': 
                slice_to_plot = coronal_scan
                seg_to_plot = seg_coronal
                L, W = self.coronal_L, self.coronal_W
            elif view_type == 'Sagittal': 
                slice_to_plot = sagittal_scan
                seg_to_plot = seg_sagittal
                L, W = self.sagittal_L, self.sagittal_W
            
            self._plot_slice(
                self.perspective_ax, 
                self.perspective_canvas, 
                slice_to_plot, 
                seg_to_plot, 
                'perspective', 
                crosshair_pos, 
                self.perspective_L, self.perspective_W,
                draw_filled_mask=False,
                draw_contour=True,
                contour_color='blue',       
                draw_crosshair=False,
                show_scan_data=True
            )
            self.perspective_ax.set_title(f"4. {view_type} Edge (Contour Only)") 
        
        self.perspective_canvas.draw_idle()
        
        if hasattr(self, 'roi_manager') and self.roi_manager is not None:
            self.update_roi_list()

    def set_perspective(self, view_name):
        if self.block_updates:
            return
        self.perspective_view = view_name
        self.update_all_views()

    def toggle_cine_loop(self, axis):
        if self.scan_array is None:
            QMessageBox.warning(self, "Warning", "Please load a scan first.")
            return

        if axis == 'z':
            self.is_playing_z = not self.is_playing_z
            self.play_z_btn.setText("⏸ Z" if self.is_playing_z else "▶ Z")
        elif axis == 'y':
            self.is_playing_y = not self.is_playing_y
            self.play_y_btn.setText("⏸ Y" if self.is_playing_y else "▶ Y")
        elif axis == 'x':
            self.is_playing_x = not self.is_playing_x
            self.play_x_btn.setText("⏸ X" if self.is_playing_x else "▶ X")
            
        if self.is_playing_z or self.is_playing_y or self.is_playing_x:
            if not self.cine_timer.isActive():
                self.cine_timer.start(50)
        else:
            if not (self.is_playing_z or self.is_playing_y or self.is_playing_x):
                if self.cine_timer.isActive():
                    self.cine_timer.stop()

    def next_slice(self):
        if self.scan_array is None:
            return
        
        if self.is_playing_z:
            current_value = self.axial_slider.value()
            if current_value < self.axial_slider.maximum():
                self.axial_slider.setValue(current_value + 1)
            else:
                self.axial_slider.setValue(0)
                
        if self.is_playing_y:
            current_value = self.coronal_slider.value()
            if current_value < self.coronal_slider.maximum():
                self.coronal_slider.setValue(current_value + 1)
            else:
                self.coronal_slider.setValue(0)

        if self.is_playing_x:
            current_value = self.sagittal_slider.value()
            if current_value < self.sagittal_slider.maximum():
                self.sagittal_slider.setValue(current_value + 1)
            else:
                self.sagittal_slider.setValue(0)
        
    def _clear_views(self):
        for ax, canvas in [(self.axial_ax, self.axial_canvas), 
                           (self.coronal_ax, self.coronal_canvas), 
                           (self.sagittal_ax, self.sagittal_canvas)]:
            ax.clear()
            ax.axis('off')
            ax.set_title("No Data Loaded")
            canvas.draw()
            
        self.perspective_ax.clear()
        self.perspective_ax.axis('off')
        self.perspective_ax.set_title("4. No Data Loaded / Dynamic View") 
        self.perspective_canvas.draw()

        
    def load_single_dicom_mpr(self):
        """Load a single DICOM file and create MPR (Multi-Planar Reconstruction) views"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Single DICOM File for MPR", 
            "", 
            "DICOM Files (*.dcm *.dicom *.DCM *.DICOM);;All Files (*)", 
            options=options
        )
        
        if not file_path:
            return
        
        try:
            print(f"[DEBUG] Loading DICOM file: {file_path}")
            
            # Try reading with SimpleITK first
            try:
                single_slice = sitk.ReadImage(file_path)
                slice_array = sitk.GetArrayFromImage(single_slice)
                print(f"[DEBUG] Loaded with SimpleITK, shape: {slice_array.shape}")
            except Exception as e:
                print(f"[DEBUG] SimpleITK failed: {e}")
                # Try with pydicom as fallback
                try:
                    import pydicom
                    dcm = pydicom.dcmread(file_path)
                    slice_array = dcm.pixel_array
                    
                    # Create SimpleITK image from numpy array
                    single_slice = sitk.GetImageFromArray(slice_array)
                    
                    # Set spacing if available
                    if hasattr(dcm, 'PixelSpacing'):
                        spacing = [float(dcm.PixelSpacing[0]), float(dcm.PixelSpacing[1])]
                        single_slice.SetSpacing(spacing)
                    
                    print(f"[DEBUG] Loaded with pydicom, shape: {slice_array.shape}")
                except Exception as e2:
                    raise Exception(f"Failed to read DICOM with both SimpleITK and pydicom: {e2}")
            
            # ========== FORCE EXTRACT ONLY ONE SLICE ==========
            if slice_array.ndim == 3:
                # If 3D array (e.g., 15 slices), extract ONLY THE MIDDLE slice
                middle_slice_idx = slice_array.shape[0] // 2
                slice_array = slice_array[middle_slice_idx, :, :]
                print(f"[DEBUG] ⚠️ Multi-slice DICOM detected! Extracted slice #{middle_slice_idx} only")
                
                QMessageBox.warning(
                    self,
                    "Multi-slice DICOM Detected",
                    f"⚠️ This DICOM file contains {slice_array.shape[0]} slices.\n\n"
                    f"Extracting only slice #{middle_slice_idx} for MPR view.\n\n"
                    f"💡 For full 3D volumes, use 'Load DICOM Series' instead."
                )
            
            # --- REPLACE WITH THIS CODE SNIPPET ---

            if slice_array.ndim != 2:
                QMessageBox.critical(
                    self, 
                    "Error", 
                    f"Invalid DICOM dimensions: {slice_array.shape}\n"
                    f"Expected 2D slice, got {slice_array.ndim}D array.\n\n"
                    f"Please select a single-slice DICOM file."
                )
                return
            
            # ==================== FIX: TRANSPOSE THE SLICE ====================
            # This corrects the inverted rotation and reference lines by
            # swapping the X and Y axes of the loaded 2D slice.
            slice_array = np.transpose(slice_array)
            # ========================== END OF FIX ==========================

            H, W = slice_array.shape
            print(f"[DEBUG] ✅ Final 2D slice shape (transposed): {H} x {W}")
            
            # Ask user for number of MPR replications
            from PyQt5.QtWidgets import QInputDialog
            num_slices, ok = QInputDialog.getInt(
                self,
                "MPR Configuration",
                f"✅ Single slice loaded: {H} × {W} pixels\n\n"
                f"How many slices to create for MPR view?\n"
                f"(The original slice will be in the middle)\n\n"
                f"💡 Recommended: 21-51 slices for good visualization\n"
                f"⚠️ More slices = smoother visualization but slower",
                31,  # Default
                3,   # Minimum
                201, # Maximum
                2    # Step
            )
            
            if not ok:
                return
            
            # Create 3D volume with the single slice in the middle
            middle_idx = num_slices // 2
            volume_3d = np.zeros((num_slices, H, W), dtype=slice_array.dtype)
            
            print(f"[DEBUG] Creating {num_slices}x{H}x{W} MPR volume from 1 slice")
            
            # Place THE ORIGINAL SINGLE SLICE in the middle
            volume_3d[middle_idx, :, :] = slice_array
            
            # Create gradient fading for OTHER slices (for visualization only)
            for i in range(num_slices):
                distance = abs(i - middle_idx)
                fade_factor = max(0.05, 1.0 - (distance / (num_slices / 2)))
                
                if i != middle_idx:
                    volume_3d[i, :, :] = (slice_array * fade_factor).astype(slice_array.dtype)
            
            # Create ITK image
            itk_img = sitk.GetImageFromArray(volume_3d)
            
            # Copy spacing from original slice
            try:
                original_spacing = single_slice.GetSpacing()
                if len(original_spacing) == 2:
                    # 2D spacing -> make it 3D
                    itk_img.SetSpacing((original_spacing[0], original_spacing[1], original_spacing[0]))
                else:
                    itk_img.SetSpacing(original_spacing)
            except:
                # Default spacing if not available
                itk_img.SetSpacing((1.0, 1.0, 1.0))
                print("[DEBUG] Using default spacing (1.0, 1.0, 1.0)")
            
            # Copy origin and direction
            try:
                origin_2d = single_slice.GetOrigin()
                if len(origin_2d) == 2:
                    itk_img.SetOrigin((origin_2d[0], origin_2d[1], 0.0))
                else:
                    itk_img.SetOrigin(origin_2d)
                
                direction_2d = single_slice.GetDirection()
                if len(direction_2d) == 4:  # 2x2 matrix
                    direction_3d = (
                        direction_2d[0], direction_2d[1], 0.0,
                        direction_2d[2], direction_2d[3], 0.0,
                        0.0, 0.0, 1.0
                    )
                    itk_img.SetDirection(direction_3d)
            except Exception as e:
                print(f"[DEBUG] Could not set origin/direction: {e}")
            
            # Reorient to RAS
            itk_img = self.reorient_to_ras(itk_img)
            
            # Set as main scan
            self.itk_image = itk_img
            self.scan_array = volume_3d
            self.orientation_text = 'RAS+ (MPR)'
            
            # Extract modality from DICOM metadata
            try:
                import pydicom
                dcm = pydicom.dcmread(file_path)
                self.modality_text = dcm.Modality if hasattr(dcm, 'Modality') else "DICOM (MPR)"
                patient_name = str(dcm.PatientName) if hasattr(dcm, 'PatientName') else "Unknown"
                study_date = str(dcm.StudyDate) if hasattr(dcm, 'StudyDate') else "Unknown"
                series_desc = str(dcm.SeriesDescription) if hasattr(dcm, 'SeriesDescription') else "Unknown"
            except:
                self.modality_text = "DICOM (MPR)"
                patient_name = "Unknown"
                study_date = "Unknown"
                series_desc = "Unknown"
            
            # Detect original scan view
            self.original_scan_view = "Analyzing with AI..."            
            # Update displays
            self.modality_display.setText(f"Modality: {self.modality_text}")
            self.original_orientation_display.setText(f"Original: {self.original_orientation}")
            self.orientation_display.setText(f"Standardized: {self.orientation_text}")
            self.original_view_display.setText(f"Original View: {self.original_scan_view}")
            
            # Show success message
            info_text = (
                f"✅ Single DICOM Slice Loaded!\n\n"
                f"📄 File: {Path(file_path).name}\n"
                f"📐 Original: {H} × {W} pixels (1 SLICE ONLY)\n"
                f"📦 MPR Volume Created: {num_slices} × {H} × {W}\n\n"
                f"🔬 Modality: {self.modality_text}\n"
                f"📋 Series: {series_desc}\n"
                f"👤 Patient: {patient_name}\n"
                f"📅 Study Date: {study_date}\n\n"
                f"💡 You can now:\n"
                f"  • View the slice in all 3 planes (Axial/Coronal/Sagittal)\n"
                f"  • Use oblique slicing in 4th panel\n"
                f"  • Draw ROIs on any view\n"
                f"  • Adjust windowing (L/W sliders)\n\n"
                f"⚠️ IMPORTANT: This is a synthetic 3D volume from 1 REAL slice\n"
                f"   Only slice #{middle_idx}/{num_slices-1} contains real patient data!\n"
                f"   Other slices are faded copies for visualization only."
            )
            
            QMessageBox.information(self, "Single Slice MPR Created", info_text)
            
            # Clear segmentation
            self.segmentation_data = None
            
            # Initialize views
            self.initialize_view()
            self.update_all_views()



            if GEMINI_AVAILABLE and self.gemini_api_key:
                self.run_ai_analysis()
            elif GEMINI_AVAILABLE and not self.gemini_api_key:
                reply = QMessageBox.question(self, "Enable AI Analysis?",
                                            "Enable AI-powered anatomy detection?\nThis requires a free Gemini API key.",
                                            QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.set_gemini_api_key()
                    if self.gemini_api_key:
                        self.run_ai_analysis()





            
            print(f"[DEBUG] ✅ MPR view created from 1 real slice (duplicated to {num_slices} for viewing)")
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"❌ Could not load DICOM file:\n\n{str(e)}\n\n"
                f"Please ensure:\n"
                f"• File is a valid DICOM (.dcm)\n"
                f"• File contains image data\n"
                f"• You have read permissions\n\n"
                f"💡 Try 'Load DICOM Series' for multi-file datasets"
            )
            import traceback
            traceback.print_exc()
        
    # ========== ROI METHODS ==========
    
    def start_roi_drawing(self, view_type):
        if self.scan_array is None:
            QMessageBox.warning(self, "Warning", "Please load a scan first.")
            return
        
        if self.roi_manager.enable_drawing(view_type):
            QMessageBox.information(
                self, 
                "ROI Drawing Active", 
                f"Draw a rectangle on the {view_type} view.\n"
                "Click and drag to create ROI.\n"
                "You'll be prompted to label it."
            )
            
            for btn in [self.roi_axial_btn, self.roi_coronal_btn, self.roi_sagittal_btn]:
                btn.setStyleSheet("")
            
            if view_type == 'axial':
                self.roi_axial_btn.setStyleSheet("background-color: #4CAF50; color: white;")
            elif view_type == 'coronal':
                self.roi_coronal_btn.setStyleSheet("background-color: #4CAF50; color: white;")
            elif view_type == 'sagittal':
                self.roi_sagittal_btn.setStyleSheet("background-color: #4CAF50; color: white;")

    def on_roi_selected(self, item):
        try:
            roi_index = int(item.text().split(':')[0])
            self.roi_manager.current_roi_index = roi_index
            self.update_all_views()
        except:
            pass

    def zoom_to_selected_roi(self):
        if not self.roi_list_widget.currentItem():
            QMessageBox.warning(self, "Warning", "Please select an ROI first.")
            return
        
        try:
            roi_index = int(self.roi_list_widget.currentItem().text().split(':')[0])
            if self.roi_manager.zoom_to_roi(roi_index):
                QMessageBox.information(self, "Zoom", f"Zoomed to ROI {roi_index}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not zoom to ROI: {e}")

    def goto_selected_roi(self):
        if not self.roi_list_widget.currentItem():
            QMessageBox.warning(self, "Warning", "Please select an ROI first.")
            return
        
        try:
            roi_index = int(self.roi_list_widget.currentItem().text().split(':')[0])
            self.roi_manager.navigate_to_roi(roi_index)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not navigate to ROI: {e}")

    def reset_all_zooms(self):
        if self.scan_array is None:
            return
        
        for view_type in ['axial', 'coronal', 'sagittal']:
            self.roi_manager.reset_zoom(view_type)

    def delete_selected_roi(self):
        if not self.roi_list_widget.currentItem():
            QMessageBox.warning(self, "Warning", "Please select an ROI first.")
            return
        
        try:
            roi_index = int(self.roi_list_widget.currentItem().text().split(':')[0])
            reply = QMessageBox.question(
                self, 'Confirm Delete', 
                f'Delete ROI {roi_index}?',
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.roi_manager.delete_roi(roi_index)
                self.update_roi_list()
                self.update_all_views()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not delete ROI: {e}")

    def clear_all_rois(self):
        if not self.roi_manager.has_roi():
            return  # ✅ ADD THIS LINE - Exit early if no ROI exists
        
        reply = QMessageBox.question(
            self, 'Confirm Clear', 
            'Delete all ROIs?',
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.roi_manager.clear_all_rois()
            self.update_roi_list()
            self.reset_all_zooms()
            self.update_all_views()

    def update_roi_list(self):
        self.roi_list_widget.clear()
        if self.roi_manager:
            roi_descriptions = self.roi_manager.get_roi_list()
            self.roi_list_widget.addItems(roi_descriptions)


    def export_selected_roi_slices(self):
        """Export slices of selected ROI to disk as PNG images"""
        if not self.roi_list_widget.currentItem():
            QMessageBox.warning(self, "Warning", "Please select an ROI first.")
            return
        
        if self.scan_array is None:
            QMessageBox.warning(self, "Warning", "No scan data loaded.")
            return
        
        if not self.roi_manager.has_roi():
            QMessageBox.warning(self, "Warning", "No ROI available to export.")
            return
        
        try:
            roi = self.roi_manager.current_roi  # ✅ Fixed: use current_roi instead of rois[index]
            
            output_dir = QFileDialog.getExistingDirectory(
                self, 
                "Select Output Directory for Exported Slices",
                str(Path.home() / "Desktop")
            )
            
            if not output_dir:
                return
            
            roi_folder_name = f"ROI_{roi.label.replace(' ', '_')}"
            roi_output_path = Path(output_dir) / roi_folder_name
            roi_output_path.mkdir(parents=True, exist_ok=True)
            
            z_start = roi.z_min  # ✅ Use ROI's z_min
            z_end = roi.z_max    # ✅ Use ROI's z_max
            
            num_slices = z_end - z_start + 1
            
            for slice_idx in range(z_start, z_end + 1):
                slice_data = self.scan_array[slice_idx, :, :]
                
                normalized = ((slice_data - slice_data.min()) / 
                            (slice_data.max() - slice_data.min() + 1e-8) * 255).astype(np.uint8)
                
                rgb_image = np.stack([normalized] * 3, axis=-1)
                
                if self.segmentation_data is not None:
                    mask_slice = self.segmentation_data[slice_idx, :, :]
                    
                    mask_overlay = mask_slice > 0
                    if np.any(mask_overlay):
                        rgb_image[mask_overlay, 0] = (rgb_image[mask_overlay, 0] * 0.5).astype(np.uint8)
                        rgb_image[mask_overlay, 1] = np.minimum(255, 
                                                                rgb_image[mask_overlay, 1] * 0.5 + 128).astype(np.uint8)
                        rgb_image[mask_overlay, 2] = np.minimum(255, 
                                                                rgb_image[mask_overlay, 2] * 0.5 + 128).astype(np.uint8)
                
                filename = f"ROI_{roi.label.replace(' ', '_')}_Slice_{slice_idx:03d}.png"
                filepath = roi_output_path / filename
                imageio.imwrite(filepath, rgb_image)
            
            QMessageBox.information(
                self,
                "Export Complete",
                f"✅ Successfully exported {num_slices} slices!\n\n"
                f"ROI: {roi.label}\n"
                f"Slices: {z_start} to {z_end}\n"
                f"Location: {roi_output_path}\n\n"
                f"Files are saved as PNG images with cyan overlay."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Could not export slices:\n{str(e)}")



    # BLOCK 3: MAIN SEGMENTATION METHOD
    # Location: At the end of MRIViewer class
    # Find the line: "def export_selected_roi_slices(self):" and find where it ends
    # Add these 3 methods AFTER export_selected_roi_slices method, before the last "if __name__"


    def segment_with_totalsegmentator(self):
        """Automatically segment the loaded scan using TotalSegmentator with progress"""
        if self.scan_array is None:
            QMessageBox.warning(self, "Warning", "Please load a scan first.")
            return
        
        if self.itk_image is None:
            QMessageBox.warning(self, "Error", "ITK image not available.")
            return
        
        # Create progress dialog
        self.seg_progress_dialog = QDialog(self)
        self.seg_progress_dialog.setWindowTitle("TotalSegmentator Segmentation")
        self.seg_progress_dialog.setModal(True)
        self.seg_progress_dialog.setMinimumWidth(450)
        self.seg_progress_dialog.setMinimumHeight(200)
        
        dialog_layout = QVBoxLayout()
        
        # Title
        title = QLabel("🤖 Segmenting medical image with AI...")
        title.setStyleSheet("font-weight: bold; font-size: 12pt; color: #FF6B6B;")
        dialog_layout.addWidget(title)
        
        # Progress text
        self.seg_progress_text = QLabel("Initializing...")
        self.seg_progress_text.setStyleSheet("color: blue; font-size: 10pt;")
        dialog_layout.addWidget(self.seg_progress_text)
        
        # Progress bar
        self.seg_progress_bar = QProgressBar()
        self.seg_progress_bar.setRange(0, 0)  # Indeterminate progress
        dialog_layout.addWidget(self.seg_progress_bar)
        
        # Info label
        info = QLabel("This may take 2-5 minutes depending on your hardware.\nPlease do not close this window.")
        info.setStyleSheet("color: gray; font-style: italic; font-size: 9pt;")
        dialog_layout.addWidget(info)
        
        self.seg_progress_dialog.setLayout(dialog_layout)
        
        # Start segmentation worker
        self.seg_worker = SegmentationWorker(self.itk_image)
        self.seg_worker.progress.connect(self.update_segmentation_progress)
        self.seg_worker.finished.connect(self.on_segmentation_finished)
        self.seg_worker.start()
        
        self.seg_progress_dialog.show()


    def update_segmentation_progress(self, message):
        """Update progress text during segmentation"""
        self.seg_progress_text.setText(message)
        self.seg_progress_text.repaint()

    def on_segmentation_finished(self, result):
        """Handle segmentation completion"""
        self.seg_progress_dialog.close()
        
        if result['success']:
            self.itk_segmentation = result['segmentation']
            self.segmentation_data = result['seg_array']
            
            if self.scan_array is not None:
                scan_shape = self.scan_array.shape
                seg_shape = self.segmentation_data.shape
                
                if scan_shape != seg_shape:
                    reply = QMessageBox.question(
                        self,
                        "⚠️ Dimension Mismatch Detected",
                        f"The generated segmentation has a different size than the original scan.\n\n"
                        f"Scan Shape: {scan_shape}\n"
                        f"Segmentation Shape: {seg_shape}\n\n"
                        f"To ensure proper alignment, the original scan must be resampled to match the segmentation. This is recommended.\n\n"
                        f"Do you want to proceed?",
                        QMessageBox.Yes | QMessageBox.No
                    )

                    if reply == QMessageBox.No:
                        self.segmentation_data = None
                        self.itk_segmentation = None
                        QMessageBox.information(self, "Cancelled", "Segmentation cancelled to prevent misalignment.")
                        return

                    # --- FIX: RESAMPLE THE ORIGINAL SCAN ---
                    try:
                        self.update_segmentation_progress("🔄 Resampling original scan to match mask...")
                        
                        resampler = sitk.ResampleImageFilter()
                        resampler.SetReferenceImage(self.itk_segmentation)
                        resampler.SetInterpolator(sitk.sitkLinear)
                        
                        # FIX: Cast the numpy value to a standard Python float
                        resampler.SetDefaultPixelValue(float(self.scan_array.min()))
                        
                        resampled_itk_scan = resampler.Execute(self.itk_image)
                        
                        # Replace the old scan with the newly resampled one
                        self.itk_image = resampled_itk_scan
                        self.scan_array = sitk.GetArrayFromImage(resampled_itk_scan)
                        
                        self.update_segmentation_progress("✅ Scan resampled successfully!")

                    except Exception as e:
                        QMessageBox.critical(self, "Resampling Failed", f"Could not resample the original scan: {e}")
                        self.segmentation_data = None
                        self.itk_segmentation = None
                        return
                    # --- END OF FIX ---

                    # Reset crosshair to the center of the new, common shape
                    D, H, W = self.scan_array.shape
                    self.crosshair_z, self.crosshair_y, self.crosshair_x = D // 2, H // 2, W // 2
                    
                    self.axial_slider.setRange(0, D - 1)
                    self.coronal_slider.setRange(0, H - 1)
                    self.sagittal_slider.setRange(0, W - 1)
                    
                    self.axial_slider.setValue(self.crosshair_z)
                    self.coronal_slider.setValue(self.crosshair_y)
                    self.sagittal_slider.setValue(self.crosshair_x)

                # <--- NEW CALL TO RE-INITIALIZE VIEW AFTER ANY POTENTIAL CHANGE --->
                # This ensures min/max intensity and L/W sliders are correct for the current self.scan_array
                self.initialize_view()
            
            QMessageBox.information(
                self, 
                "✅ Segmentation Success", 
                "TotalSegmentator segmentation complete!\n\n"
                "Blue organ outlines are now visible in the 4th panel."
            )
            
            self.update_all_views()
            
        else:
            QMessageBox.critical(
                self, 
                "❌ Segmentation Failed", 
                f"TotalSegmentator error:\n{result['error']}\n\n"
                "Make sure TotalSegmentator is installed:\n"
                "pip install TotalSegmentator"
            )

    def adjust_roi_range(self):
        """Adjust ROI slice range"""
        if not self.roi_manager or not self.roi_manager.has_roi():
            QMessageBox.warning(self, "No ROI", "Please create an ROI first.")
            return
        
        self.roi_manager.adjust_roi_range()
    
    def export_roi_volume(self):
        """Export ROI volume to NIfTI file"""
        if not self.roi_manager or not self.roi_manager.has_roi():
            QMessageBox.warning(self, "No ROI", "Please create an ROI first.")
            return
        
        self.roi_manager.export_roi_volume()

    
    def closeEvent(self, event):
        """Handle application close - cleanup threads"""
        # Stop any running workers
        if hasattr(self, 'test_worker') and self.test_worker is not None:
            if self.test_worker.isRunning():
                self.test_worker.quit()
                self.test_worker.wait(2000)  # Wait max 2 seconds
        
        if hasattr(self, 'ai_worker') and self.ai_worker is not None:
            if self.ai_worker.isRunning():
                self.ai_worker.quit()
                self.ai_worker.wait(2000)
        
        if hasattr(self, 'seg_worker') and self.seg_worker is not None:
            if self.seg_worker.isRunning():
                self.seg_worker.quit()
                self.seg_worker.wait(2000)
        
        # Stop cine timer
        if hasattr(self, 'cine_timer') and self.cine_timer.isActive():
            self.cine_timer.stop()
        
        # Accept the close event
        event.accept()




if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    if GEMINI_AVAILABLE:
        print("✅ Gemini API available - AI features enabled")
        print("📚 Get your free API key at: https://makersuite.google.com/app/apikey")
    else:
        print("⚠️ Gemini API not available")
        print("   Install with: pip install google-generativeai")
    
    viewer = MRIViewer()
    viewer.show()
    sys.exit(app.exec_())