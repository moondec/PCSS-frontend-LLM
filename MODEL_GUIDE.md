# PCSS LLM Model Guide / Przewodnik po Modelach PCSS

This document provides a summary of models available in the application, categorized by their use cases and strengths.
*Ten dokument zawiera zestawienie modeli dostępnych w aplikacji, z podziałem na ich zastosowania i mocne strony.*

---

## 🇬🇧 English Version

### 🇵🇱 Polish Models (Specialized)
Best for Polish language, culture, and grammar tasks.

*   **Bielik-11b (v2)**
    *   **Architecture:** SpeakLeash (based on Solar/Mistral).
    *   **Best for:** Official letters, emails in Polish, summarizing Polish texts, tasks requiring correct inflection.
    *   **Note:** The "default" model for Polish tasks.

*   **Bielik-4.5b**
    *   **Architecture:** Smaller version of Bielik.
    *   **Best for:** Quick responses, simple translations, running on lower-end hardware (if local).

### 🧠 General Purpose Giants
Powerful models with general knowledge, comparable to GPT-4.

*   **DeepSeek-V3 / V3.1**
    *   **Strengths:** Logic, mathematics, coding, very long context.
    *   **Best for:** Solving reasoning puzzles, analyzing long documents, writing code.

*   **GPT-4o (OpenAI)**
    *   **Availability:** Currently **NOT AVAILABLE** on PCSS (use for text tasks only if available in list).
    *   **Note:** Multi-modal features (vision) are disabled.

*   **Llama 3.3 (70B)**
    *   **Maker:** Meta.
    *   **Strengths:** Solid general model, great writing style.
    *   **Best for:** Content generation in English and Polish, brainstorming, general assistance.

*   **Qwen2.5 (72B)**
    *   **Maker:** Alibaba.
    *   **Strengths:** Often tops Open Source leaderboards. Excellent in math and coding.
    *   **Best for:** Complex instructions, STEM tasks.

### 💻 Coding
Models trained specifically to understand programming languages.

*   **Qwen3-Coder (30B)**
    *   **Specialization:** Programming.
    *   **Best for:** Writing scripts (Python, JS, C++), debugging, explaining code. Often outperforms general 70B models at code.

*   **Mistral-Small (24B)**
    *   **Strengths:** Speed and efficiency. Great quality-to-speed ratio.
    *   **Best for:** Quick scripts, refactoring, simple technical questions.

### ⚕️ Medicine & Science (Specialized)
Models with specialized domain knowledge.

*   **Meditron3:70b**
    *   **Specialization:** Medicine.
    *   **Best for:** Answering medical questions, analyzing clinical cases, summarizing medical literature.
    *   **Warning:** For educational/research purposes only, does not replace a doctor.

*   **OpenBioLLM:70b**
    *   **Specialization:** Biology and biomedicine.
    *   **Best for:** Working with scientific publications in biology, genetics, and pharmacy.

### 🛠️ Tools

*   **Nanonets-OCR-s**
    *   **Type:** OCR (Optical Character Recognition).
    *   **Use:** Not a chatbot. Extracts text from images, scans, and PDF files without a text layer.

*   **gpt-oss_120b / 20b**
    *   **Type:** Experimental/Internal PCSS models.
    *   **Use:** Likely large open-source models (e.g., Falcon or Mixtral) for testing. Worth trying if others fail at specific tasks.

---

## 🇵🇱 Wersja Polska

### 🇵🇱 Modele Polskie (Specjalizowane)
Te modele najlepiej radzą sobie z językiem polskim, naszą kulturą i gramatyką.

*   **Bielik-11b (v2)**
    *   **Architektura:** SpeakLeash (bazujący na Solar/Mistral).
    *   **Najlepsze do:** Pisania pism urzędowych, e-maili po polsku, streszczania polskich tekstów, zadań wymagających poprawnej odmiany fleksyjnej.
    *   **Uwagi:** Model "domyślny" dla zadań w języku polskim.

*   **Bielik-4.5b**
    *   **Architektura:** Mniejsza wersja Bielika.
    *   **Najlepsze do:** Szybkich odpowiedzi, prostych tłumaczeń, działania na słabszym sprzęcie (gdyby był uruchamiany lokalnie).

### 🧠 Wszechstronne Giganty (General Purpose)
Najpotężniejsze modele o ogólnej wiedzy, porównywalne z GPT-4.

*   **DeepSeek-V3 / V3.1**
    *   **Mocne strony:** Logika, matematyka, programowanie, bardzo długi kontekst.
    *   **Najlepsze do:** Rozwiązywania zagadek logicznych, analizy długich dokumentów, pisania kodu.

*   **GPT-4o (OpenAI)**
    *   **Dostępność:** Obecnie **NIEDOSTĘPNY** na PCSS (używaj do zadań tekstowych tylko jeśli jest na liście).
    *   **Uwaga:** Funkcje multimodalne (wizja) są wyłączone.

*   **Llama 3.3 (70B)**
    *   **Producent:** Meta.
    *   **Mocne strony:** Bardzo solidny model ogólny, świetny styl wypowiedzi.
    *   **Najlepsze do:** Generowania treści po angielsku i polsku, burze mózgów, asystent ogólny.

*   **Qwen2.5 (72B)**
    *   **Producent:** Alibaba.
    *   **Mocne strony:** Często wygrywa rankingi Open Source. Świetny w matematyce i kodowaniu.
    *   **Najlepsze do:** Skomplikowanych instrukcji, zadań ścisłych.

### 💻 Programowanie i Kod (Coding)
Modele wytrenowane specjalnie do rozumienia języków programowania.

*   **Qwen3-Coder (30B)**
    *   **Specjalizacja:** Programowanie.
    *   **Najlepsze do:** Pisania skryptów (Python, JS, C++), debugowania, wyjaśniania kodu. Radzi sobie lepiej z kodem niż ogólne modele 70B.

*   **Mistral-Small (24B)**
    *   **Mocne strony:** Szybkość i efektywność. Bardzo dobry stosunek jakości do prędkości.
    *   **Najlepsze do:** Szybkich skryptów, refaktoryzacji, prostych pytań technicznych.

### ⚕️ Medycyna i Nauka (Specialized)
Modele posiadające specjalistyczną wiedzę dziedzinową.

*   **Meditron3:70b**
    *   **Specjalizacja:** Medycyna.
    *   **Najlepsze do:** Odpowiadania na pytania medyczne, analizy przypadków klinicznych, streszczania literatury medycznej.
    *   **Ostrzeżenie:** Służy do celów edukacyjnych/badawczych, nie zastępuje lekarza.

*   **OpenBioLLM:70b**
    *   **Specjalizacja:** Biologia i biomedycyna.
    *   **Najlepsze do:** Pracy z publikacjami naukowymi z zakresu biologii, genetyki i farmacji.

### 🛠️ Narzędzia

*   **Nanonets-OCR-s**
    *   **Typ:** OCR (Optical Character Recognition).
    *   **Zastosowanie:** To nie jest chatbot. Służy do wyciągania tekstu ze zdjęć, skanów dokumentów i plików PDF, które nie mają warstwy tekstowej.

*   **gpt-oss_120b / 20b**
    *   **Typ:** Modele eksperymentalne/wewnętrzne PCSS.
    *   **Zastosowanie:** Prawdopodobnie duże modele open-source (np. Falcon lub Mixtral) udostępnione testowo. Warto sprawdzić, jeśli inne modele nie dają rady w specyficznych zadaniach.
