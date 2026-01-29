import sys
import os

# Add project root to path if running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication
from pcss_llm_app.ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # Optional: Set styling
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
