import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                             QLineEdit, QTextEdit, QComboBox, QFileDialog, 
                             QMessageBox, QDialog, QFormLayout, QListWidget)
from PySide6.QtCore import Qt, QThread, Signal, QEvent
import datetime
import markdown
import time

from pcss_llm_app.config import ConfigManager
from pcss_llm_app.core.api_client import PcssApiClient
from pcss_llm_app.core.database import DatabaseManager
from pcss_llm_app.core.file_manager import FileManager

class SettingsDialog(QDialog):
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config = config_manager
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)
        
        layout = QFormLayout()
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        # Pre-fill if exists (optional, mostly empty for security)
        if self.config.get_api_key():
             self.api_key_input.setPlaceholderText("Stored in Keyring")
        
        layout.addRow("PCSS API Key:", self.api_key_input)
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_settings)
        layout.addRow(save_btn)
        
        self.setLayout(layout)

    def save_settings(self):
        key = self.api_key_input.text().strip()
        if key:
            if self.config.set_api_key(key):
                QMessageBox.information(self, "Success", "API Key saved to Keyring.")
                self.accept()
            else:
                QMessageBox.critical(self, "Error", "Failed to save to Keyring.")
        else:
             self.accept()

class ChatWorker(QThread):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, api_client, model, messages):
        super().__init__()
        self.api_client = api_client
        self.model = model
        self.messages = messages

    def run(self):
        try:
            response = self.api_client.chat_completion(
                model=self.model,
                messages=self.messages
            )
            content = response.choices[0].message.content
            self.finished.emit(content)
        except Exception as e:
            self.error.emit(str(e))

class AgentWorker(QThread):
    """
    Worker to handle Agent interactions: Add Message -> Run -> Poll -> Fetch Response
    """
    finished = Signal(str)
    status_update = Signal(str)
    error = Signal(str)

    def __init__(self, api_client, thread_id, assistant_id, content=None):
        super().__init__()
        self.api_client = api_client
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.content = content

    def run(self):
        try:
            # 1. Add Message if content provided
            if self.content:
                self.status_update.emit("Adding message...")
                self.api_client.add_message_to_thread(self.thread_id, self.content)
            
            # 2. Start Run
            self.status_update.emit("Starting run...")
            run = self.api_client.run_thread(self.thread_id, self.assistant_id)
            run_id = run.id
            
            # 3. Poll
            while True:
                run_status = self.api_client.get_run_status(self.thread_id, run_id)
                status = run_status.status
                self.status_update.emit(f"Status: {status}")
                
                if status == 'completed':
                    break
                elif status in ['failed', 'cancelled', 'expired']:
                    self.error.emit(f"Run ended with status: {status}")
                    return
                
                time.sleep(1) # Poll interval
            
            # 4. Get items
            self.status_update.emit("Fetching response...")
            messages = self.api_client.get_thread_messages(self.thread_id)
            # Assuming the last message is from assistant (returned desc)
            if messages.data:
                latest_msg = messages.data[0]
                if latest_msg.role == "assistant":
                    # Extract text content
                    response_text = ""
                    for content_block in latest_msg.content:
                        if hasattr(content_block, 'text'):
                             response_text += content_block.text.value
                    self.finished.emit(response_text)
                else:
                    self.finished.emit("(No new assistant message found)")
            else:
                self.finished.emit("(No messages found)")
                
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCSS LLM Client (Bielik)")
        self.resize(1000, 700)
        
        self.config = ConfigManager()
        self.api = PcssApiClient(self.config)
        self.db = DatabaseManager()
        
        self.current_conversation_id = None
        self.chat_history = [] 

        # Agent State
        self.current_assistant_id = None
        self.current_thread_id = None

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
        self.history_list.itemClicked.connect(self.load_history_conversation)
        sidebar_layout.addWidget(QLabel("History"))
        sidebar_layout.addWidget(self.history_list)
        
        new_chat_btn = QPushButton("New Chat")
        new_chat_btn.clicked.connect(self.start_new_chat)
        sidebar_layout.addWidget(new_chat_btn)
        
        refresh_btn = QPushButton("Refresh History")
        refresh_btn.clicked.connect(self.refresh_history)
        sidebar_layout.addWidget(refresh_btn)

        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self.open_settings)
        sidebar_layout.addWidget(settings_btn)

        main_layout.addWidget(sidebar)

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
        
        main_layout.addWidget(self.tabs)
        
        self.refresh_history()

    def _init_chat_tab(self):
        layout = QVBoxLayout(self.chat_tab)
        
        # Controls
        controls_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        controls_layout.addWidget(QLabel("Model:"))
        controls_layout.addWidget(self.model_combo)
        
        save_btn = QPushButton("Save to File")
        save_btn.clicked.connect(self.save_to_file)
        controls_layout.addWidget(save_btn)
        
        load_btn = QPushButton("Load from File")
        load_btn.clicked.connect(self.load_from_file)
        controls_layout.addWidget(load_btn)
        
        layout.addLayout(controls_layout)
        
        # Chat Display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
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
        layout = QVBoxLayout(self.agent_tab)
        
        # Config Area
        config_layout = QHBoxLayout()
        
        self.agent_name_input = QLineEdit()
        self.agent_name_input.setPlaceholderText("Agent Name")
        config_layout.addWidget(self.agent_name_input)
        
        self.agent_instr_input = QLineEdit()
        self.agent_instr_input.setPlaceholderText("Instructions (e.g. You are a helpful bot)")
        config_layout.addWidget(self.agent_instr_input)
        
        create_agent_btn = QPushButton("Create Assistant")
        create_agent_btn.clicked.connect(self.create_assistant)
        config_layout.addWidget(create_agent_btn)
        
        create_thread_btn = QPushButton("New Thread")
        create_thread_btn.clicked.connect(self.create_thread)
        config_layout.addWidget(create_thread_btn)

        layout.addLayout(config_layout)

        # Agent Chat Display
        self.agent_display = QTextEdit()
        self.agent_display.setReadOnly(True)
        layout.addWidget(self.agent_display)
        
        # Status
        self.agent_status_label = QLabel("Ready")
        layout.addWidget(self.agent_status_label)
        
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
        
        layout.addLayout(input_layout)

    def _refresh_models(self):
        if self.api.is_configured():
            models = self.api.list_models()
            self.model_combo.clear()
            if not models:
                self.model_combo.addItem("gpt-4o") 
                self.model_combo.addItem("bielik-11b-v2.3-instruct")
            else:
                self.model_combo.addItems(models)

    def open_settings(self):
        dlg = SettingsDialog(self.config, self)
        if dlg.exec():
            self.api = PcssApiClient(self.config)
            self._refresh_models()

    def start_new_chat(self):
        self.current_conversation_id = None
        self.chat_history = []
        self.chat_display.clear()
        self.model_combo.setEnabled(True)

    def send_message(self):
        text = self.message_input.toPlainText().strip()
        if not text:
            return

        model = self.model_combo.currentText()
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

    # --- Agent Mode Methods ---
    def create_assistant(self):
        name = self.agent_name_input.text()
        instr = self.agent_instr_input.text()
        if not name or not instr:
            QMessageBox.warning(self, "Input", "Name and Instructions required.")
            return
            
        try:
            self.agent_status_label.setText("Creating Assistant...")
            # Use current model combo as the model for assistant
            model = self.model_combo.currentText()
            assistant = self.api.create_assistant(name, instr, model)
            self.current_assistant_id = assistant.id
            self.agent_status_label.setText(f"Assistant Created: {assistant.id}")
            self.agent_display.append(f"<b>System:</b> Assistant '{name}' created ({assistant.id})<br>")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.agent_status_label.setText("Error creating assistant")

    def create_thread(self):
        try:
            self.agent_status_label.setText("Creating Thread...")
            thread = self.api.create_thread()
            self.current_thread_id = thread.id
            self.agent_status_label.setText(f"Thread: {thread.id}")
            self.agent_display.clear()
            self.agent_display.append(f"<b>System:</b> New Thread started ({thread.id})<br>")
            
            # Start new agent conversation in DB as well? 
            # For simplicity, agent chats are just volatile or I could store them if I mapped thread_id to db.
            # Keeping it simple as per requirements (agent mode functionality).
        except Exception as e:
             QMessageBox.critical(self, "Error", str(e))

    def send_to_agent(self):
        if not self.current_assistant_id or not self.current_thread_id:
            QMessageBox.warning(self, "Agent", "Please create Assistant and Thread first.")
            return

        text = self.agent_input.toPlainText().strip()
        if not text:
            return
            
        html = markdown.markdown(text)
        self.agent_display.append(f"<b>User:</b> {html}<br>")
        self.agent_input.clear()
        
        self.agent_worker = AgentWorker(self.api, self.current_thread_id, self.current_assistant_id, text)
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
