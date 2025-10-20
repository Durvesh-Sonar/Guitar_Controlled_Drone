# enhanced_prototype_taskbar_control.py
import sys
import os
import time
import numpy as np
import cv2
import mss
import pyautogui
from PyQt5 import QtWidgets, QtCore, QtGui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1  # Small pause between actions

# ----------------------
# CONFIG
# ----------------------
TEMPLATES_DIR = "templates"  # folder with string images
THRESHOLD = 0.6  # Increased threshold for better accuracy
USE_EDGES = False

# Mapping: string -> app name (based on your template files)
STRING_TO_APP = {
    "E4": "whatsapp",
    "A2": "word", 
    "D3": "edge",
    "B3": "whatsapp",  # Additional mapping
    "G3": "word",      # Additional mapping
    "E2": "edge"       # Additional mapping
}

# Mapping: app -> screen coords (taskbar icons)
# Your current coordinates - verified from your code
APP_COORDS = {
    "whatsapp": (1269, 1052),
    "word": (1332, 1039),
    "edge": (1375, 1054)
}

# App display names for better UI feedback
APP_DISPLAY_NAMES = {
    "whatsapp": "WhatsApp",
    "word": "Microsoft Word",
    "edge": "Microsoft Edge"
}

# ----------------------
# Enhanced Utils
# ----------------------
def preprocess(img):
    """Enhanced preprocessing for better OCR accuracy"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Apply bilateral filter to reduce noise while preserving edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Enhance contrast
    gray = cv2.equalizeHist(gray)
    
    # Optional edge detection
    if USE_EDGES:
        gray = cv2.Canny(gray, 50, 150)
    
    return gray

def load_templates(folder):
    """Load and preprocess template images"""
    templates = {}
    print("📂 Loading templates from:", folder)
    
    if not os.path.exists(folder):
        print(f"❌ Templates folder '{folder}' not found!")
        return templates

    for fname in sorted(os.listdir(folder)):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
            
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"⚠️ Skipped (not an image): {fname}")
            continue

        # Preprocess template the same way as input
        img = cv2.bilateralFilter(img, 9, 75, 75)
        img = cv2.equalizeHist(img)
        
        if USE_EDGES:
            img = cv2.Canny(img, 50, 150)

        templates[name] = img
        print(f"✅ Loaded template: {name} ({fname})")

    print("🔎 Templates loaded:", list(templates.keys()))
    return templates

def match_symbol(frame_gray, templates):
    """Enhanced template matching with multiple scale detection"""
    best_name = None
    best_score = -1.0
    best_location = None
    
    for name, tmpl in templates.items():
        # Try multiple scales for better matching
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        
        for scale in scales:
            # Resize template
            width = int(tmpl.shape[1] * scale)
            height = int(tmpl.shape[0] * scale)
            
            # Skip if template becomes too large
            if height > frame_gray.shape[0] or width > frame_gray.shape[1]:
                continue
                
            tmpl_resized = cv2.resize(tmpl, (width, height), interpolation=cv2.INTER_AREA)
            
            # Template matching
            res = cv2.matchTemplate(frame_gray, tmpl_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_score:
                best_score = max_val
                best_name = name
                best_location = max_loc

    if best_score >= THRESHOLD:
        return best_name, best_score, best_location
    return None, best_score, None

def click_app_safely(app_name):
    """Safely click on taskbar app with error handling"""
    if app_name not in APP_COORDS:
        print(f"❌ No coordinates found for app: {app_name}")
        return False
    
    x, y = APP_COORDS[app_name]
    
    try:
        # Move mouse smoothly to target
        pyautogui.moveTo(x, y, duration=0.2)
        time.sleep(0.1)
        
        # Click the taskbar icon
        pyautogui.click(x, y)
        
        print(f"✔️ Successfully clicked {APP_DISPLAY_NAMES.get(app_name, app_name)} at ({x}, {y})")
        return True
        
    except pyautogui.FailSafeException:
        print("⚠️ PyAutoGUI failsafe triggered — mouse moved to corner.")
        return False
    except Exception as e:
        print(f"❌ Error clicking {app_name}: {str(e)}")
        return False

# ----------------------
# Enhanced PyQt5 GUI
# ----------------------
class MainWindow(QtWidgets.QWidget):
    def __init__(self, templates):
        super().__init__()
        self.setWindowTitle("🎸 Guitar OCR → Taskbar Apps")
        self.resize(600, 350)
        self.setStyleSheet("""
            QWidget { 
                background-color: #2b2b2b; 
                color: #ffffff; 
                font-family: 'Segoe UI'; 
                font-size: 11px;
            }
            QPushButton { 
                background-color: #404040; 
                border: 1px solid #555555; 
                padding: 8px 16px; 
                border-radius: 4px; 
            }
            QPushButton:hover { 
                background-color: #505050; 
            }
            QPushButton:pressed { 
                background-color: #353535; 
            }
            QPushButton:disabled { 
                background-color: #2a2a2a; 
                color: #666666; 
            }
            QLabel { 
                padding: 4px; 
            }
        """)

        # Create layout
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title = QtWidgets.QLabel("🎸 Guitar String App Launcher")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Control buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("🎵 Start Detection")
        self.stop_btn = QtWidgets.QPushButton("⏹️ Stop Detection")
        self.stop_btn.setEnabled(False)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)
        
        # Test buttons for coordinates
        test_layout = QtWidgets.QHBoxLayout()
        self.test_whatsapp_btn = QtWidgets.QPushButton("Test WhatsApp")
        self.test_word_btn = QtWidgets.QPushButton("Test Word") 
        self.test_edge_btn = QtWidgets.QPushButton("Test Edge")
        self.get_coords_btn = QtWidgets.QPushButton("Get Mouse Position")
        
        test_layout.addWidget(self.test_whatsapp_btn)
        test_layout.addWidget(self.test_word_btn)
        test_layout.addWidget(self.test_edge_btn)
        test_layout.addWidget(self.get_coords_btn)
        layout.addLayout(test_layout)
        
        # Status display
        status_group = QtWidgets.QGroupBox("Detection Status")
        status_layout = QtWidgets.QVBoxLayout(status_group)
        
        self.detection_label = QtWidgets.QLabel("Status: Ready to detect")
        self.confidence_label = QtWidgets.QLabel("Confidence: -")
        self.app_label = QtWidgets.QLabel("Target App: -")
        self.last_action_label = QtWidgets.QLabel("Last Action: -")
        
        status_layout.addWidget(self.detection_label)
        status_layout.addWidget(self.confidence_label)
        status_layout.addWidget(self.app_label)
        status_layout.addWidget(self.last_action_label)
        layout.addWidget(status_group)
        
        # String to app mapping display
        mapping_group = QtWidgets.QGroupBox("String → App Mapping")
        mapping_layout = QtWidgets.QVBoxLayout(mapping_group)
        
        for string, app in STRING_TO_APP.items():
            mapping_text = f"🎼 {string} → {APP_DISPLAY_NAMES.get(app, app)}"
            mapping_layout.addWidget(QtWidgets.QLabel(mapping_text))
        
        layout.addWidget(mapping_group)
        
        # Settings
        settings_group = QtWidgets.QGroupBox("Settings")
        settings_layout = QtWidgets.QFormLayout(settings_group)
        
        self.threshold_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.threshold_slider.setRange(30, 90)
        self.threshold_slider.setValue(int(THRESHOLD * 100))
        self.threshold_label = QtWidgets.QLabel(f"{THRESHOLD:.2f}")
        
        threshold_layout = QtWidgets.QHBoxLayout()
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        
        settings_layout.addRow("Detection Threshold:", threshold_layout)
        layout.addWidget(settings_group)
        
        # Initialize detection system
        self.templates = templates
        self.region = {"left": 915, "top": 331, "width": 100, "height": 100}
        self.sct = mss.mss()
        
        # Debounce settings
        self.last_triggered_symbol = None
        self.last_trigger_time = 0.0
        self.trigger_cooldown = 2.0  # 2 seconds between triggers
        
        # Timer for detection
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.capture_and_match)
        
        # Connect signals
        self.start_btn.clicked.connect(self.start_capture)
        self.stop_btn.clicked.connect(self.stop_capture)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        
        # Connect test buttons
        self.test_whatsapp_btn.clicked.connect(lambda: self.test_coordinates("whatsapp"))
        self.test_word_btn.clicked.connect(lambda: self.test_coordinates("word"))
        self.test_edge_btn.clicked.connect(lambda: self.test_coordinates("edge"))
        self.get_coords_btn.clicked.connect(self.get_mouse_position)

    def test_coordinates(self, app_name):
        """Test clicking coordinates for a specific app"""
        print(f"🧪 Testing coordinates for {app_name}...")
        success = click_app_safely(app_name)
        if success:
            self.last_action_label.setText(f"Test: ✅ {APP_DISPLAY_NAMES.get(app_name, app_name)} clicked successfully")
        else:
            self.last_action_label.setText(f"Test: ❌ Failed to click {app_name}")
    
    def get_mouse_position(self):
        """Get current mouse position for coordinate setup"""
        QtWidgets.QMessageBox.information(
            self, 
            "Get Mouse Position", 
            "Move your mouse to the app icon and wait 3 seconds..."
        )
        
        def get_pos():
            time.sleep(3)
            x, y = pyautogui.position()
            print(f"📍 Mouse position: ({x}, {y})")
            self.last_action_label.setText(f"Mouse Position: ({x}, {y})")
        
        # Run in separate thread to avoid blocking UI
        import threading
        threading.Thread(target=get_pos, daemon=True).start()

    def update_threshold(self, value):
        """Update detection threshold"""
        global THRESHOLD
        THRESHOLD = value / 100.0
        self.threshold_label.setText(f"{THRESHOLD:.2f}")

    def start_capture(self):
        """Start the detection process"""
        self.timer.start(500)  # Check every 500ms
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.detection_label.setText("Status: 🔍 Detecting...")
        print("🎸 Detection started!")

    def stop_capture(self):
        """Stop the detection process"""
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.detection_label.setText("Status: ⏸️ Stopped")
        print("⏹️ Detection stopped!")

    def capture_and_match(self):
        """Capture screen region and match guitar strings"""
        try:
            # Capture screen region
            bbox = self.region
            screenshot = self.sct.grab(bbox)
            frame = np.array(screenshot)
            
            if frame.shape[2] == 4:  # Convert BGRA to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Preprocess frame
            gray = preprocess(frame)

            # Debug windows (comment out in production)
            cv2.imshow("Guitar OCR - Raw", frame)
            cv2.imshow("Guitar OCR - Processed", gray)
            cv2.waitKey(1)

            # Match against templates
            symbol, score, location = match_symbol(gray, self.templates)
            
            # DEBUG: Print all detection attempts
            print(f"🔍 Detection attempt - Best match: {symbol}, Score: {score:.3f}, Threshold: {THRESHOLD:.3f}")
            
            # Update UI
            current_time = time.time()
            
            if symbol:
                self.detection_label.setText(f"Status: 🎯 Detected {symbol}")
                self.confidence_label.setText(f"Confidence: {score:.2%}")
                
                app = STRING_TO_APP.get(symbol)
                print(f"📱 String {symbol} maps to app: {app}")
                
                if app:
                    self.app_label.setText(f"Target App: {APP_DISPLAY_NAMES.get(app, app)}")
                    
                    # Check debounce
                    should_trigger = (
                        symbol != self.last_triggered_symbol or 
                        current_time - self.last_trigger_time > self.trigger_cooldown
                    )
                    
                    print(f"🕒 Should trigger? {should_trigger} (last: {self.last_triggered_symbol}, cooldown: {current_time - self.last_trigger_time:.1f}s)")
                    
                    if should_trigger:
                        print(f"[{time.strftime('%H:%M:%S')}] 🎸 TRIGGERING: {symbol} → {app} (confidence: {score:.2%})")
                        
                        # Attempt to click the app
                        success = click_app_safely(app)
                        print(f"🖱️ Click result: {success}")
                        
                        if success:
                            self.last_action_label.setText(f"Last Action: ✅ Opened {APP_DISPLAY_NAMES.get(app, app)}")
                        else:
                            self.last_action_label.setText(f"Last Action: ❌ Failed to open {app}")
                        
                        # Update debounce state
                        self.last_triggered_symbol = symbol
                        self.last_trigger_time = current_time
                    else:
                        self.last_action_label.setText(f"Last Action: ⏳ Cooldown active")
                else:
                    self.app_label.setText("Target App: No mapping found")
            else:
                self.detection_label.setText("Status: 🔍 No match found")
                self.confidence_label.setText(f"Best Score: {score:.2%} (below threshold {THRESHOLD:.0%})")
                self.app_label.setText("Target App: -")
                
        except Exception as e:
            print(f"❌ Error in capture_and_match: {str(e)}")
            self.detection_label.setText(f"Status: ❌ Error - {str(e)}")

    def closeEvent(self, event):
        """Clean up when closing the application"""
        cv2.destroyAllWindows()
        event.accept()

# ----------------------
# Main
# ----------------------
def main():
    """Main application entry point"""
    print("🎸 Starting Guitar OCR App Launcher...")
    
    # Load templates
    templates = load_templates(TEMPLATES_DIR)
    
    if not templates:
        print("❌ No templates found! Please check your templates directory.")
        return
    
    # Create and run the application
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for better dark theme
    
    main_window = MainWindow(templates)
    main_window.show()
    
    print("🚀 Application ready!")
    print("📍 Current app coordinates:")
    for app_name, coords in APP_COORDS.items():
        print(f"   {APP_DISPLAY_NAMES.get(app_name, app_name)}: {coords}")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()