# PCSS LLM Client (Bielik)

<img src="resources/logo.png" align="right" width="240" />

A Python desktop application (GUI) for interacting with the PCSS LLM Service, built with **PySide6 (Qt)** and **LangChain**.

## ✨ Key Features

### 1. 💬 Chat Mode
-   **Conversation History**: All chats are saved locally to an SQLite database (`conversations.db`).
-   **Model Selection**: Dynamically fetches models from PCSS (e.g., `bielik-11b-v2.3-instruct`, `gpt-4o`).
-   **Import/Export**: Save and load specific conversations to JSON files.
-   **Markdown Support**: Full rendering of headings, code blocks, and lists.

### 2. 🤖 Agent Mode (Autonomous)
The application features a powerful Agent capable of performing tasks on your local file system.
-   **Workspace Security**: The agent is strictly confined to a specific directory (configurable in settings).
-   **File Tools**: The agent can autonomously using tools:
    -   `ListDirectory`: See files.
    -   `ReadFile`: Read content of files.
    -   `WriteFile`: Create or overwrite files.
    -   `Copy/Move/Delete`: Manage files.
-   **Use Case**: "Read this report.pdf and create a summary.txt" -> Agent does it automatically.
-   **Autonomous Agent**: A "Agent Mode" that can perform complex tasks (File management, text processing).
-   **Vision Capabilities**: Hybrid agent using GPT-4o for image analysis (`analyze_image`).
-   **Internet Access**: Integrated **DuckDuckGo Search** for real-time information retrieval.
-   **Document Generation**: Create professional PDF/DOCX reports (via Pandoc).
-   **Local & Secure**: Your data stays on your machine (except for API calls to PCSS/OpenAI).

### 3. 🔒 Security
-   **Secure Storage**: API Keys are stored in the system Keyring (macOS Keychain, Windows Credential Locker), never in plain text.
-   **Local Data**: All history and settings are stored locally.
-   **Documentation**: See [MODEL_GUIDE.md](MODEL_GUIDE.md) and [TOOLS_GUIDE.md](TOOLS_GUIDE.md).

### 4. 🤖 Chat Mode vs. Agent Mode (Important!)
The application has two distinct tabs:

| Feature | **Chat Tab** | **Agent Mode Tab** |
| :--- | :--- | :--- |
| **Primary Use** | Conversation, Brainstorming, Q&A | **Executing Tasks**, File Operations, Research |
| **Tools Access** | ❌ No Tools | ✅ **Has Tools** (Internet, Vision, Files, Pandoc) |
| **Internet** | ❌ Offline (Knowledge cutoff) | ✅ **Online** (via DuckDuckGo) |
| **Vision** | ❌ Text only | ✅ **Vision** (via GPT-4o proxy) |

> [!IMPORTANT]
> If you want the model to **search the web**, **read files**, or **analyze images**, you MUST use the **Agent Mode** tab. The standard Chat tab is for pure text conversation only.

## 🛠️ Installation

### Prerequisites
-   **Anaconda** or **Miniconda** installed.
-   Python 3.10+
-   **Pandoc** >= 3.0 ([Download](https://github.com/jgm/pandoc/releases)) - Required for document conversion.

### Setup Pipeline

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/moondec/PCSS-frontend-LLM.git
    cd Bielik
    ```

2.  **Create Environment**
    You can use the provided `environment.yml`:
    ```bash
    conda env create -f environment.yml
    conda activate bielik
    ```

    *Or manually:*
    ```bash
    conda create -n bielik python=3.10 -y
    conda activate bielik
    pip install pyside6 openai keyring markdown langchain langchain-openai langchain-community pypdf python-docx pypandoc ddgs
    ```

## ⚙️ Configuration

1.  **API Key**: On first launch, enter your PCSS Cloud API Token. It corresponds to your active Grant.
2.  **Workspace**: In **Settings**, select the directory where the Agent is allowed to work (Default: `~/Documents/Bielik_Workspace`).

## ▶️ Usage

```bash
# Activate environment
conda activate bielik

# Run
python pcss_llm_app/main.py
```

### Tips
-   **Chat**: Use `Shift+Enter` for new lines, `Enter` to send.
-   **Agent**: To initialize a session, go to "Agent Mode" -> "Create Assistant" (this boots the LangChain engine). Then type requests like "Create a python script hello.py in my workspace".

## 🏗️ Technology Stack
-   **GUI**: PySide6 (Qt)
-   **LLM Engine**: LangGraph / LangChain
-   **API**: OpenAI Compatible (PCSS)
-   **Database**: SQLite
