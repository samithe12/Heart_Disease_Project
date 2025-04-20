# Heart Disease Prediction - BCS 222 Programming Paradigms

## Course: BCS 222 – Programming Paradigms
Instructor: Dr. Haythem El-Messiry

## Team Members
* Sami Aftab (20230003344)
* Marwan Mohammud (20240004471)
* Shehab Hatem (20220002769)
* Abdulrahman Fuzan (20230003508)

## Project Goal
To develop an AI model using Python to predict the presence of heart disease based on the UCI Cleveland dataset. This project specifically explores the integration of **Procedural**, **Object-Oriented (OOP)**, and **Functional (FP)** programming paradigms as required by the course.

## Current Status (as of Sun, Apr 20th Evening)

* **Core Implementation Complete!**
* Project setup complete (Repo structure, venv, `requirements.txt`, `.gitignore`).
* Dataset (`processed.cleveland.data`) loaded via `data_loader.py`.
* Exploratory Data Analysis (EDA) performed (`Data_Exploration.ipynb`) - key insights identified (missing values in `ca`/`thal`, types, target balance ~54%/46%, feature relationships).
* **Object-Oriented Programming (OOP):** Implemented `DataPreprocessor` class (`data_preprocessor.py`) encapsulating the preprocessing pipeline (Imputation, Scaling, Encoding) using Scikit-learn's `Pipeline` and `ColumnTransformer`.
* **Functional Programming (FP):** Implemented pure functions (`evaluation_metrics.py`) for calculating accuracy, precision, recall, and F1-score.
* **Procedural Programming:** Orchestrated the complete workflow (Load -> Preprocess -> Split -> Train -> Evaluate) in the main script (`run_pipeline.py`).
* **Model Training & Evaluation:** Successfully trained and evaluated a `LogisticRegression` model on the preprocessed data using a train/test split. Evaluation metrics were calculated using the custom functional metrics functions, yielding reasonable results (e.g., ~88.5% accuracy). (Random Forest was initially evaluated but removed for simplification).

## Repository Structure
* `data_base/`: Contains the `processed.cleveland.data` file.
* `Data_Exploration.ipynb`: Notebook with EDA steps and findings.
* `data_loader.py`: Script/module for loading data.
* `data_preprocessor.py`: Contains the OOP `DataPreprocessor` class.
* `evaluation_metrics.py`: Contains the pure functions for metrics (FP Demo).
* `run_pipeline.py`: Main script executing the workflow (Procedural Demo).
* `requirements.txt`: Lists required Python packages and exact versions.
* `.gitignore`: Specifies files/folders for Git to ignore (like `.venv`).
* `README.md`: This file.

## Getting Started (Setup for Teammates)

**IMPORTANT:** Follow these steps carefully in VS Code to ensure everyone has the same working environment.

1.  **Clone the Repository:**
    * Open VS Code.
    * Use the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`) and type `Git: Clone`.
    * Paste the repository URL: `[Sami: PASTE THE GITHUB REPO URL HERE]` and choose a location on your computer to save the project folder.
    * Once cloned, open the project folder in VS Code (`File -> Open Folder...`).

2.  **Open the Integrated Terminal in VS Code:**
    * Go to the menu `Terminal -> New Terminal` (or `Ctrl+\``).
    * Make sure the terminal is inside the project folder (`Heart_Disease_Project`).

3.  **Create Your Virtual Environment:**
    * In the VS Code terminal, run:
        ```bash
        python -m venv .venv
        ```
        *(Use `python3` if needed on your system. This creates a `.venv` folder inside the project - it's already in `.gitignore` so it won't be committed).*

4.  **Activate the Virtual Environment:**
    * **In the SAME VS Code terminal**, run the activation command for your OS:
        * **Windows (Command Prompt/PowerShell):**
            ```cmd
            .\.venv\Scripts\activate
            ```
        * **macOS / Linux (bash/zsh):**
            ```bash
            source .venv/bin/activate
            ```
    * You should see `(.venv)` appear at the start of your terminal prompt.
    * *VS Code Hint:* VS Code might ask if you want to select this environment for the workspace - click Yes.

5.  **Install Required Packages:**
    * While the environment is active (`(.venv)` visible), run:
        ```bash
        pip install -r requirements.txt
        ```
    * This installs the exact same package versions used during setup.

6.  **You're Ready!** You can now run the notebooks (`.ipynb`) and scripts (`.py`) within VS Code, using the activated virtual environment.

## Next Steps & Roles (Final Deliverables Focus)

**Deadline: Tuesday, April 22nd - Final Push!**

The core coding is complete. The remaining focus is entirely on finalizing the deliverables:

1.  **Report Writing (Highest Priority - Lead: Abdulrahman, Input: All):**
    * **Goal:** Complete the comprehensive final report.
    * **Tasks:** Compile all sections (Intro, Problem, EDA Summary [from Sami], Solution Design [Proc/OOP/FP justifications - input from Marwan/Shehab/Sami], Implementation Details, Evaluation Results [LogReg metrics], **Paradigm Analysis** [Pros/Cons - team reflection], Conclusion, References).
    * **Critical:** Ensure the Design and Analysis sections clearly explain *how* and *why* the paradigms were used and evaluate their effectiveness in *this* project.

2.  **Presentation Creation (Lead: Abdulrahman):**
    * **Goal:** Create slides summarizing the report.
    * **Tasks:** Cover key aspects: Problem, Data, Design/Paradigms, Implementation, Results, Analysis, Conclusion.

3.  **Code Cleanup, Comments & Final Commit (Responsibility: All):**
    * **Goal:** Ensure final code is readable, documented, and submitted correctly.
    * **Tasks:** Review all `.py` and `.ipynb` files. Add comments (`#`) explaining logic clearly. Ensure consistent formatting. Push final commented version to GitHub.

4.  **Q&A Preparation (Responsibility: All):**
    * **Goal:** Be ready for individual questions.
    * **Tasks:** Review the final code, report draft, and presentation slides.

5.  **Collaboration:**
    * Provide all necessary information (EDA summary, design details, FP details) to Abdulrahman promptly.
    * Use GitHub for final code push.
    * Coordinate via group chat/Google Doc.

