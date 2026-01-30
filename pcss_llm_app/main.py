import sys
import os

# Add project root to path if running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from pcss_llm_app.ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # Optional: Set styling
    app.setStyle("Fusion")

    # Set App Icon Global
    logo_path = os.path.join(os.path.dirname(__file__), "..", "resources", "logo.png")
    if os.path.exists(logo_path):
        app.setWindowIcon(QIcon(logo_path))

    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
