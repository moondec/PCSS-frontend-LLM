import sys
import os
import glob
import yaml
from PySide6.QtCore import Qt, QSize, QTimer, QThread, Signal, QObject, QEvent
from PySide6.QtGui import QAction, QIcon, QTextCursor, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTextEdit, QTextBrowser, QLineEdit, QPushButton, QLabel, 
    QComboBox, QSplitter, QFrame, QScrollArea, QSizePolicy,
    QApplication, QTabWidget, QFileDialog, QMessageBox, QDialog, QFormLayout, QListWidget, QMenu
)
import datetime
import markdown
import time

from pcss_llm_app.config import ConfigManager
from pcss_llm_app.core.api_client import PcssApiClient
from pcss_llm_app.core.database import DatabaseManager

from pcss_llm_app.core.file_manager import FileManager
from pcss_llm_app.core.agent_engine import LangChainAgentEngine

class AgentLogSignal(QObject):
    log_message = Signal(str)

class SettingsDialog(QDialog):
    def __init__(self, config_manager, parent=None, available_models=None):
        super().__init__(parent)
        self.config = config_manager
        self.available_models = available_models or []
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)
        

        settings_dlg_layout = QFormLayout()
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        # Pre-fill if exists (optional, mostly empty for security)
        if self.config.get_api_key():
             self.api_key_input.setPlaceholderText("Stored in Keyring")
        
        settings_dlg_layout.addRow("PCSS API Key:", self.api_key_input)
        
        # Workspace Path
        self.workspace_input = QLineEdit(self.config.get_workspace_path())
        workspace_layout = QHBoxLayout()
        workspace_layout.addWidget(self.workspace_input)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_workspace)
        workspace_layout.addWidget(browse_btn)
        settings_dlg_layout.addRow("Workspace:", workspace_layout)

        # Model Selection
        self.model_combo = QComboBox()
        if self.available_models:
            self.model_combo.addItems(self.available_models)
        else:
            self.model_combo.addItems(["gpt-4o", "bielik_11b"]) # Default fallback
            
        current_model = self.config.get("model", "gpt-4o")
        self.model_combo.setCurrentText(current_model)
        settings_dlg_layout.addRow("Default Model:", self.model_combo)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_settings)
        settings_dlg_layout.addRow(save_btn)
        
        self.setLayout(settings_dlg_layout)

    def browse_workspace(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Workspace Directory")
        if dir_path:
            self.workspace_input.setText(dir_path)

    def save_settings(self):
        api_key = self.api_key_input.text().strip()
        if api_key:
            if not self.config.set_api_key(api_key):
                QMessageBox.critical(self, "Error", "Failed to save API Key to Keyring.")
                return # Don't proceed if API key save failed
            
        workspace_path = self.workspace_input.text().strip()
        if workspace_path:
            self.config.set_workspace_path(workspace_path)
            
        selected_model = self.model_combo.currentText()
        if selected_model:
            self.config.set("model", selected_model)
            
        self.accept()

class ChatWorker(QThread):
    finished = Signal(str)
    error = Signal(str)
    log_message = Signal(str)

    def __init__(self, api_client, model, messages):
        super().__init__()
        self.api_client = api_client
        self.model = model
        self.messages = messages

    def run(self):
        try:
            self.log_message.emit(f"ChatWorker: Sending request to model '{self.model}'...")
            self.log_message.emit(f"ChatWorker: Input messages: {len(self.messages)}")
            
            response = self.api_client.chat_completion(
                model=self.model,
                messages=self.messages
            )
            content = response.choices[0].message.content
            self.log_message.emit("ChatWorker: Response received.")
            self.log_message.emit(f"ChatWorker: Response length: {len(content)} chars.")
            
            self.finished.emit(content)
        except Exception as e:
            self.log_message.emit(f"ChatWorker Error: {str(e)}")
            self.error.emit(str(e))

class AgentWorker(QThread):
    """
    Worker to handle Agent interactions via LangChain Engine
    """
    finished = Signal(str)
    status_update = Signal(str)
    error = Signal(str)

    def __init__(self, agent_engine, text, chat_history=None):
        super().__init__()
        self.agent_engine = agent_engine
        self.text = text
        self.chat_history = chat_history if chat_history else []

    def run(self):
        try:
            self.status_update.emit("Agent thinking...")
            # Run the agent (synchronous call in thread)
            response = self.agent_engine.run(self.text, self.chat_history)
            self.finished.emit(response)
                
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCSS LLM Client")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set App Icon
        logo_path = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "logo.png")
        if os.path.exists(logo_path):
             self.setWindowIcon(QIcon(logo_path))

        # Main Layout      
        self.config = ConfigManager()
        self.api = PcssApiClient(self.config)
        self.db = DatabaseManager()
        
        self.current_conversation_id = None
        self.current_conversation_id = None
        self.current_agent_conversation_id = None
        self.chat_history = [] 

        # Agent Log Signal
        self.agent_logger = AgentLogSignal()
        self.agent_logger.log_message.connect(self.append_log)

        # Agent State (LangChain)
        self.agent_engine = None
        self.agent_history = []

        self._init_ui()
        
        # Check API Key
        if not self.config.get_api_key():
            QMessageBox.warning(self, "Setup", "Please configure your API Key in Settings.")
            self.open_settings()
        
        # Try to load models
        self._refresh_models()
    
    def eventFilter(self, source, event):
        if event.type() == QEvent.KeyPress:
             if hasattr(self, 'message_input') and source is self.message_input and \
                event.key() in [Qt.Key_Return, Qt.Key_Enter] and not (event.modifiers() & Qt.ShiftModifier):
                 self.send_message()
                 return True
             if hasattr(self, 'agent_input') and source is self.agent_input and \
                event.key() in [Qt.Key_Return, Qt.Key_Enter] and not (event.modifiers() & Qt.ShiftModifier):
                 self.send_to_agent()
                 return True
        return super().eventFilter(source, event)

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Sidebar (History)
        sidebar = QWidget()
        sidebar.setFixedWidth(200)
        sidebar_layout = QVBoxLayout(sidebar)
        
        self.history_list = QListWidget()
        self.history_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.history_list.customContextMenuRequested.connect(self.show_history_context_menu)
        self.history_list.itemClicked.connect(self.load_history_conversation)
        sidebar_layout.addWidget(QLabel("History"))
        sidebar_layout.addWidget(self.history_list)
        
        new_chat_btn = QPushButton("New Chat")
        new_chat_btn.clicked.connect(self.start_new_chat)
        sidebar_layout.addWidget(new_chat_btn)
        
        refresh_btn = QPushButton("Refresh History")
        refresh_btn.clicked.connect(self.refresh_history)
        sidebar_layout.addWidget(refresh_btn)

        clear_btn = QPushButton("Clear All History")
        clear_btn.clicked.connect(self.clear_history)
        sidebar_layout.addWidget(clear_btn)
        sidebar_layout.addWidget(refresh_btn)

        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self.open_settings)
        sidebar_layout.addWidget(settings_btn)

        main_layout.addWidget(sidebar)

        # Main Content Area (Splitter for Tabs + Console)
        self.main_content_splitter = QSplitter(Qt.Vertical)
        
        # Tab Widget
        self.tabs = QTabWidget()
        
        # Chat Tab
        self.chat_tab = QWidget()
        self._init_chat_tab()
        self.tabs.addTab(self.chat_tab, "Chat")
        
        # Agent Tab
        self.agent_tab = QWidget()
        self._init_agent_tab()
        self.tabs.addTab(self.agent_tab, "Agent Mode")
        
        self.main_content_splitter.addWidget(self.tabs)
        
        # Global Console
        self.console_display = QTextEdit()
        self.console_display.setReadOnly(True)
        self.console_display.setPlaceholderText("Debug Console Log...")
        self.console_display.setStyleSheet("background-color: #2b2b2b; color: #00ff00; font-family: monospace;")
        self.console_display.hide()
        self.main_content_splitter.addWidget(self.console_display)
        self.main_content_splitter.setSizes([600, 150])

        main_layout.addWidget(self.main_content_splitter)
        
        self.refresh_history()

    def _init_chat_tab(self):
        layout = QVBoxLayout(self.chat_tab)
        
        # Controls
        controls_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        controls_layout.addWidget(QLabel("Model:"))
        controls_layout.addWidget(self.model_combo)
        
        save_btn = QPushButton("Save to File")
        save_btn.clicked.connect(self.save_to_file)
        controls_layout.addWidget(save_btn)
        
        load_btn = QPushButton("Load from File")
        load_btn.clicked.connect(self.load_from_file)
        controls_layout.addWidget(load_btn)

        # Chat Console Toggle
        self.toggle_console_chat_btn = QPushButton("Show Debug Console")
        self.toggle_console_chat_btn.setCheckable(True)
        self.toggle_console_chat_btn.toggled.connect(self.toggle_console)
        controls_layout.addWidget(self.toggle_console_chat_btn)
        
        layout.addLayout(controls_layout)
        
        # Chat Display
        self.chat_display = QTextBrowser()
        self.chat_display.setReadOnly(True)
        self.chat_display.setOpenExternalLinks(True)
        layout.addWidget(self.chat_display)
        
        # Input
        input_layout = QHBoxLayout()
        self.message_input = QTextEdit()
        self.message_input.setFixedHeight(80)
        self.message_input.installEventFilter(self)
        input_layout.addWidget(self.message_input)
        
        send_btn = QPushButton("Send")
        send_btn.setFixedSize(80, 80)
        send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(send_btn)
        
        layout.addLayout(input_layout)

    def _init_agent_tab(self):
        agent_layout = QVBoxLayout(self.agent_tab) # Renamed to avoid conflict with 'layout'
        
        # Config Area - Row 1: Name and Profile
        config_layout = QHBoxLayout()
        
        self.agent_name_input = QLineEdit()
        self.agent_name_input.setPlaceholderText("Agent Name")
        self.agent_name_input.setFixedWidth(150)
        config_layout.addWidget(self.agent_name_input)
        
        # Profile Selection
        config_layout.addWidget(QLabel("Profile:"))
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(200)
        self.profile_combo.currentTextChanged.connect(self._on_profile_changed)
        config_layout.addWidget(self.profile_combo)
        
        refresh_profiles_btn = QPushButton("↻")
        refresh_profiles_btn.setFixedWidth(30)
        refresh_profiles_btn.setToolTip("Refresh Profiles")
        refresh_profiles_btn.clicked.connect(self._load_agent_profiles)
        config_layout.addWidget(refresh_profiles_btn)
        
        open_folder_btn = QPushButton("📁")
        open_folder_btn.setFixedWidth(30)
        open_folder_btn.setToolTip("Open Profiles Folder")
        open_folder_btn.clicked.connect(self._open_profiles_folder)
        config_layout.addWidget(open_folder_btn)
        
        config_layout.addStretch()
        
        create_agent_btn = QPushButton("Create Assistant")
        create_agent_btn.clicked.connect(self.create_assistant)
        config_layout.addWidget(create_agent_btn)
        
        create_thread_btn = QPushButton("New Thread")
        create_thread_btn.clicked.connect(self.create_thread)
        config_layout.addWidget(create_thread_btn)

        agent_layout.addLayout(config_layout)
        
        # Load profiles on init
        self.agent_profiles = {}  # {name: {description, instructions}}
        self.current_profile_instructions = ""
        self._load_agent_profiles()

        # Agent Chat Display
        self.agent_display = QTextBrowser()
        self.agent_display.setReadOnly(True)
        self.agent_display.setOpenExternalLinks(True)
        agent_layout.addWidget(self.agent_display)

        # Agent Console Toggle
        self.toggle_console_agent_btn = QPushButton("Show Debug Console")
        self.toggle_console_agent_btn.setCheckable(True)
        self.toggle_console_agent_btn.toggled.connect(self.toggle_console)
        config_layout.addWidget(self.toggle_console_agent_btn)
        
        # Status Area
        status_layout = QHBoxLayout()
        self.agent_status_label = QLabel("Status: Idle")
        workspace_path = self.config.get_workspace_path()
        self.workspace_label = QLabel(f"Workspace: {workspace_path}")
        self.workspace_label.setStyleSheet("color: gray; font-size: 10px;")
        
        status_layout.addWidget(self.agent_status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.workspace_label)
        
        agent_layout.addLayout(status_layout)
        
        # Input Area
        input_layout = QHBoxLayout()
        self.agent_input = QTextEdit()
        self.agent_input.setFixedHeight(80)
        self.agent_input.installEventFilter(self)
        input_layout.addWidget(self.agent_input)
        
        send_btn = QPushButton("Send to Agent")
        send_btn.setFixedSize(100, 80)
        send_btn.clicked.connect(self.send_to_agent)
        input_layout.addWidget(send_btn)
        
        agent_layout.addLayout(input_layout)

    def _refresh_models(self):
        if self.api.is_configured():
            self.model_combo.blockSignals(True) # Prevent triggering change reset during update
            models = self.api.list_models()
            self.model_combo.clear()
            if not models:
                self.model_combo.addItem("gpt-4o") 
                self.model_combo.addItem("bielik_11b")
            else:
                self.model_combo.addItems(models)
            self.model_combo.blockSignals(False)

    def on_model_changed(self, text):
        if not text:
            return
            
        print(f"DEBUG: Model changed to {text}")        
        # Reset Chat
        self.start_new_chat()
        
        # Reset Agent
        self.agent_history = []
        self.current_agent_conversation_id = None
        self.agent_display.clear()
        self.agent_display.append(f"<b>System:</b> Model changed to {text}. Please re-initialize Assistant.<br>")
        self.agent_engine = None # Force re-creation with new model
        self.agent_status_label.setText("Model Changed")

    def open_settings(self):
        # Get current models from main combo
        current_models = [self.model_combo.itemText(i) for i in range(self.model_combo.count())]
        
        dlg = SettingsDialog(self.config, self, available_models=current_models)
        if dlg.exec():
            self.api = PcssApiClient(self.config)
            self._refresh_models()

    def start_new_chat(self):
        self.current_conversation_id = None
        self.chat_history = []
        self.chat_display.clear()
        self.model_combo.setEnabled(True)

    def append_log(self, message: str):
        self.console_display.append(message)
        # Auto scroll
        cursor = self.console_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.console_display.setTextCursor(cursor)

    def send_message(self):
        text = self.message_input.toPlainText().strip()
        if not text:
            return

        model = self.model_combo.currentText()
        print(f"DEBUG: Sending message with model: {model}")
        if not self.current_conversation_id:
            title = text[:30] + "..."
            self.current_conversation_id = self.db.create_conversation(title, model)
            self.refresh_history()

        self._append_message("User", text)
        self.message_input.clear()
        
        self.chat_history.append({"role": "user", "content": text})
        self.db.add_message(self.current_conversation_id, "user", text)

        self.worker = ChatWorker(self.api, model, self.chat_history)
        self.worker.finished.connect(self.handle_response)
        self.worker.error.connect(self.handle_error)
        self.worker.log_message.connect(self.append_log) # Connect logging
        self.worker.start()
        
        self.message_input.setEnabled(False)

    def handle_response(self, content):
        self._append_message("AI", content)
        self.chat_history.append({"role": "assistant", "content": content})
        self.db.add_message(self.current_conversation_id, "assistant", content)
        self.message_input.setEnabled(True)
        self.message_input.setFocus()

    def handle_error(self, err_msg):
        QMessageBox.critical(self, "API Error", err_msg)
        self.message_input.setEnabled(True)

    def _append_message(self, role, text):
        html = markdown.markdown(text)
        self.chat_display.append(f"<b>{role}:</b> {html}<br>")

    # --- Agent Profile Methods ---
    def _load_agent_profiles(self):
        """Load agent profiles from agent_profiles/ directory"""
        profiles_dir = os.path.join(os.path.dirname(__file__), "..", "agent_profiles")
        profiles_dir = os.path.abspath(profiles_dir)
        
        self.agent_profiles = {}
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        
        # Add "No Profile" option
        self.profile_combo.addItem("(No Profile)")
        self.agent_profiles["(No Profile)"] = {"description": "No custom instructions", "instructions": ""}
        
        if os.path.exists(profiles_dir):
            for yaml_file in glob.glob(os.path.join(profiles_dir, "*.yaml")):
                try:
                    with open(yaml_file, 'r', encoding='utf-8') as f:
                        profile = yaml.safe_load(f)
                        if profile and 'name' in profile:
                            name = profile['name']
                            self.agent_profiles[name] = {
                                'description': profile.get('description', ''),
                                'instructions': profile.get('instructions', '')
                            }
                            self.profile_combo.addItem(name)
                except Exception as e:
                    print(f"Error loading profile {yaml_file}: {e}")
        
        self.profile_combo.blockSignals(False)
        
        # Select first real profile if available
        if self.profile_combo.count() > 1:
            self.profile_combo.setCurrentIndex(1)
    
    def _on_profile_changed(self, profile_name):
        """Handle profile selection change"""
        if profile_name in self.agent_profiles:
            profile = self.agent_profiles[profile_name]
            self.current_profile_instructions = profile.get('instructions', '')
            # Update tooltip with description
            self.profile_combo.setToolTip(profile.get('description', ''))
        else:
            self.current_profile_instructions = ""
    
    def _open_profiles_folder(self):
        """Open agent_profiles folder in file manager"""
        profiles_dir = os.path.join(os.path.dirname(__file__), "..", "agent_profiles")
        profiles_dir = os.path.abspath(profiles_dir)
        if os.path.exists(profiles_dir):
            os.system(f'open "{profiles_dir}"')  # macOS
        else:
            QMessageBox.warning(self, "Error", f"Profiles directory not found: {profiles_dir}")

    # --- Agent Mode Methods ---
    def create_assistant(self):
        name = self.agent_name_input.text() or "Assistant"
        # Get instructions from selected profile
        instructions = self.current_profile_instructions
        profile_name = self.profile_combo.currentText()
        
        # Create/Init Engine
        api_key = self.config.get_api_key()
        if not api_key:
            QMessageBox.warning(self, "Error", "API Key not set.")
            return

        workspace = self.config.get_workspace_path()
        model = self.model_combo.currentText()
        
        try:
            self.agent_status_label.setText("Initializing Agent...")
            # Initialize engine with profile instructions
            self.agent_engine = LangChainAgentEngine(
                api_key, model, workspace, 
                log_callback=self.agent_logger.log_message.emit,
                custom_instructions=instructions
            )
            
            self.agent_status_label.setText("Agent Ready")
            self.agent_history = [] # Reset history
            
            # Start new persistence session for Agent
            self.current_agent_conversation_id = None # Will be created on first message
            
            self.agent_display.append(f"<b>System:</b> Agent '{name}' initialized with profile: {profile_name}<br>")
            self.agent_display.append(f"<b>System:</b> Workspace: {workspace}<br>")
            self.agent_display.append(f"<b>System:</b> Tools: [Files, Documents, OCR, Vision, Web Search]<br>")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.agent_status_label.setText("Init Error")

    def create_thread(self):
        # In LangChain mode, "New Thread" just clears memory
        self.agent_history = []
        self.current_agent_conversation_id = None
        self.agent_display.clear()
        self.agent_display.append("<b>System:</b> Memory cleared. New Session.<br>")
        self.agent_status_label.setText("Ready")

    def send_to_agent(self):
        if not self.agent_engine:
            QMessageBox.warning(self, "Agent", "Please Initialize Agent first (Create Assistant button).")
            return

        text = self.agent_input.toPlainText().strip()
        if not text:
            return
            
        html = markdown.markdown(text)
        self.agent_display.append(f"<b>User:</b> {html}<br>")
        self.agent_input.clear()
        
        # Add to history
        # (LangChain agent handles history internally if passed? My engine run takes history)
        # We pass self.agent_history? 
        # Actually my engine implementation takes list of messages.
        # But AgentExecutor with chat_history handles it.
        # We need to maintain the list of (human, ai) tuples or BaseMessages.
        # Let's assume engine.run expects a list.
        
        # Persistence
        if not self.current_agent_conversation_id:
             model = self.model_combo.currentText()
             title = f"Agent: {text[:20]}..."
             self.current_agent_conversation_id = self.db.create_conversation(title, model, mode="agent")
             self.refresh_history()
        
        self.db.add_message(self.current_agent_conversation_id, "user", text)
        
        self.agent_worker = AgentWorker(self.agent_engine, text, self.agent_history)
        self.agent_status_label.setText("Processing...")
        self.agent_worker.status_update.connect(self.update_agent_status)
        self.agent_worker.finished.connect(self.handle_agent_response)
        self.agent_worker.error.connect(self.handle_agent_error)
        self.agent_input.setEnabled(False)
        self.agent_worker.start()

    def update_agent_status(self, status):
        self.agent_status_label.setText(status)

    def handle_agent_response(self, content):
        html = markdown.markdown(content)
        self.agent_display.append(f"<b>Agent:</b> {html}<br>")
        self.agent_status_label.setText("Ready")
        self.agent_input.setEnabled(True)
        self.agent_input.setFocus()
        
        # Update history
        # Since I'm managing history manually to pass to agent, I should append here.
        # Although AgentExecutor returns output, it doesn't return the updated "chat_history" list automatically unless configured.
        # I need to append input and output to self.agent_history.
        # Input was self.agent_worker.text (need to store it or access it)
        input_text = self.agent_worker.text
        
        from langchain_core.messages import HumanMessage, AIMessage
        self.agent_history.append(HumanMessage(content=input_text))
        self.agent_history.append(AIMessage(content=content))
        self.agent_status_label.setText("Ready")
        self.agent_input.setEnabled(True)
        self.agent_input.setFocus()

        self.agent_input.setEnabled(True)
        self.agent_input.setFocus()
        
        # Persist Agent Response
        if self.current_agent_conversation_id:
            self.db.add_message(self.current_agent_conversation_id, "assistant", content)

    def toggle_console(self, checked):
        # Sync buttons
        self.toggle_console_chat_btn.blockSignals(True)
        self.toggle_console_agent_btn.blockSignals(True)
        
        self.toggle_console_chat_btn.setChecked(checked)
        self.toggle_console_agent_btn.setChecked(checked)
        
        text = "Hide Debug Console" if checked else "Show Debug Console"
        self.toggle_console_chat_btn.setText(text)
        self.toggle_console_agent_btn.setText(text)
        
        self.toggle_console_chat_btn.blockSignals(False)
        self.toggle_console_agent_btn.blockSignals(False)

        if checked:
            self.console_display.show()
        else:
            self.console_display.hide()

    def handle_agent_error(self, err):
        QMessageBox.critical(self, "Agent Error", err)
        self.agent_status_label.setText("Error")
        self.agent_input.setEnabled(True)

    # --- Common Methods ---
    def refresh_history(self):
        self.history_list.clear()
        rows = self.db.get_conversations()
        for r in rows:
            item_text = f"{r[1]} ({r[2]})"
            self.history_list.addItem(item_text)
            self.history_list.item(self.history_list.count()-1).setData(Qt.UserRole, r[0])

    def load_history_conversation(self, item):
        conv_id = item.data(Qt.UserRole)
        self.current_conversation_id = conv_id
        
        messages = self.db.get_messages(conv_id)
        
        self.chat_display.clear()
        self.chat_history = []
        
        for role, content, _ in messages:
            self._append_message("AI" if role == "assistant" else "User", content)
            self.chat_history.append({"role": role, "content": content})

    def show_history_context_menu(self, position):
        item = self.history_list.itemAt(position)
        if not item:
            return
            
        menu = QMenu()
        delete_action = menu.addAction("Delete Conversation")
        action = menu.exec(self.history_list.mapToGlobal(position))
        
        if action == delete_action:
            conv_id = item.data(Qt.UserRole)
            self.delete_history_item(conv_id)

    def delete_history_item(self, conv_id):
        reply = QMessageBox.question(self, "Confirm Delete", 
                                   "Are you sure you want to delete this conversation?",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.db.delete_conversation(conv_id)
            if self.current_conversation_id == conv_id:
                self.start_new_chat()
            self.refresh_history()

    def clear_history(self):
        reply = QMessageBox.question(self, "Confirm Clear All", 
                                   "Are you sure you want to delete ALL history? This cannot be undone.",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.db.clear_all_conversations()
            self.start_new_chat()
            self.refresh_history()

    def save_to_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Conversation", "", "JSON Files (*.json)")
        if path:
            data = {
                "meta": {
                    "model": self.model_combo.currentText(),
                    "date": str(datetime.datetime.now())
                },
                "messages": self.chat_history
            }
            if FileManager.save_conversation(path, data):
                QMessageBox.information(self, "Saved", "Conversation saved successfully.")

    def load_from_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Conversation", "", "JSON Files (*.json)")
        if path:
            data = FileManager.load_conversation(path)
            if data:
                self.start_new_chat()
                if "messages" in data:
                    for msg in data["messages"]:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        self._append_message("AI" if role == "assistant" else "User", content)
                        self.chat_history.append({"role": role, "content": content})
