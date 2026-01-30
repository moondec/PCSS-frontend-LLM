# Agent Tools Guide

This document describes the tools available to the Autonomous Agent in the PCSS LLM Client. The agent automatically selects the best tool for the job.

## 📂 File Management
Basic file system operations within the workspace.
*   **list_directory**: Lists files and folders.
*   **read_file**: Reads proper contents of text files.
*   **write_file**: Creates or overwrites text files.
*   **copy_file / move_file / delete_file**: Standard file operations.

## 📄 Document Processing (Advanced)
Specialized tools for handling office documents.
*   **read_docx / read_pdf**: Extracts text from DOCX and PDF files.
*   **write_docx**: Creates simple Word documents (plain text only).
*   **convert_document**:
    *   *Function:* Converts files between formats (e.g., HTML -> DOCX, HTML -> PDF).
    *   *Usage Strategy:* To create a complex report, the Agent first writes an HTML file (with tables, headers, bold text) and then converts it to DOCX/PDF.

## 👁️ Vision & OCR (Hybrid Agent)
Tools that allow the agent to "see" images using vision models.

*   **analyze_image** (Hybrid Vision):
    *   *Model:* **GPT-4o** (always).
    *   *Function:* Describes images, analyzes charts, understands document layouts, and answers questions about visual content.
    *   *Why Hybrid?* Even if your main agent is "Bielik" (text-only), this tool delegates the visual part to GPT-4o, giving you the best of both worlds.

*   **ocr_image**:
    *   *Model:* **Nanonets-OCR-s**.
    *   *Function:* Strictly extracts text from scans/images. Prefer `analyze_image` for understanding context.

## 🤖 How to use?
Just ask the agent!
*   *"Read this invoice.png and tell me the total."* (Uses `analyze_image`)
*   *"Create a sales report PDF with a table."* (Uses `write_file` [html] -> `convert_document`)
