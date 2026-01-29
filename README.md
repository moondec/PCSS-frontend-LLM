# PCSS LLM Client (Bielik)

A Python desktop application (GUI) for interacting with the PCSS LLM Service, built with **PySide6 (Qt)** and **OpenAI SDK**. 

Features:
- **Chat Interface**: Standard chat with history saved to local SQLite database.
- **Agent Mode**: Interface for OpenAI Assistants API (Create Assistants, Threads, Runs).
- **Secure Configuration**: API tokens are stored securely in the system Keyring (macOS Keychain, etc.), never in plain text.
- **Model Selection**: Dynamically fetches available models from the API.

## 🛠️ Prerequisites

- **Anaconda** or **Miniconda** installed on your system.
- macOS (recommended for Keyring integration, though libraries are cross-platform).

## 🚀 Installation Pipeline

### 1. Clone the Repository
```bash
git clone <repository_url>
cd Bielik
```

### 2. Prepare the Environment
We use Conda to manage dependencies. You can create the environment manually or using the provided `environment.yml`.

**Option A: Using environment.yml (Recommended)**
```bash
conda env create -f environment.yml
conda activate bielik
```

**Option B: Manual Setup**
```bash
# Create environment with Python 3.10
conda create -n bielik python=3.10 -y

# Activate environment
conda activate bielik

# Install dependencies (using pip inside conda is often smoother for PySide6/OpenAI updates)
pip install pyside6 openai keyring markdown
```

## ⚙️ Configuration

The application requires a PCSS Cloud Grant API Key.

1.  Obtain your API Key from [PCSS Cloud](https://pcss.plcloud.pl).
2.  Run the application.
3.  On first launch, a **Settings** dialog will appear.
4.  Paste your API Key and click **Save**.
    -   *Note: The key is saved securely to your OS Keyring Service. It is NOT stored in any text file in this repository.*

## ▶️ Running the Application

```bash
# Ensure environment is active
conda activate bielik

# Run the entry point
python pcss_llm_app/main.py
```

## 📂 Project Structure

- `pcss_llm_app/`: Main package.
    - `main.py`: Entry point.
    - `ui/`: Qt Widgets and Windows.
    - `core/`: Logic layer.
        - `api_client.py`: OpenAI SDK wrapper for PCSS.
        - `database.py`: SQLite handling.
        - `config.py`: Keyring and settings management.
- `conversations.db`: Local database (created on first run).

## 📝 License
[Choose a license, e.g., MIT]
