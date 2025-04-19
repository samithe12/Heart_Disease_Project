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

## Current Status (as of Saturday, Apr 19th Evening)
* Initial project structure created locally.
* Virtual environment (`venv`) set up using Python's built-in module.
* Required Python packages installed (`requirements.txt` generated).
* Dataset (`processed.cleveland.data`) downloaded and placed in `data_base/` folder.
* Initial Exploratory Data Analysis (EDA) completed in `Data_Exploration.ipynb`:
    * Data loaded and inspected successfully.
    * Target variable analyzed, converted to binary (0=No Disease, 1=Disease Present), and found to be reasonably balanced (~54% vs 46%).
    * Numerical features analyzed (`.describe()`, histograms, box plots); outliers noted in `chol` and `oldpeak`.
    * Key categorical features analyzed visually (`countplot` with `hue=target_binary`), identifying potentially strong predictors (`sex`, `cp`, `exang`, `slope`, `ca`, `thal`).
* Placeholder Python files created (e.g., `data_loader.py`, `data_preprocessor.py`, etc. - *Sami: Adjust names if you used the alternatives we discussed*).
* `.gitignore` file created to exclude `.venv` and `__pycache__`.

## Repository Structure
* `data_base/`: Contains the `processed.cleveland.data` file.
* `Data_Exploration.ipynb`: Jupyter Notebook with EDA steps and findings.
* `data_loader.py`: Script/module for loading data.
* `data_preprocessor.py`: Placeholder for the OOP preprocessing pipeline class.
* `evaluation_metrics.py`: Placeholder for functional evaluation metric functions.
* `custom_transforms.py`: Placeholder for any custom functional transformations.
* `run_pipeline.py`: Placeholder for the main script orchestrating the process.
* `requirements.txt`: Lists required Python packages and exact versions.
* `.gitignore`: Specifies files/folders for Git to ignore.
* `README.md`: This file.

## Getting Started (Setup for Teammates)

**IMPORTANT:** Follow these steps carefully in VS Code to ensure everyone has the same working environment.

1.  **Clone the Repository:**
    * Open VS Code.
    * Open the Command Palette (`Ctrl+Shift+P` on Windows/Linux, `Cmd+Shift+P` on Mac).
    * Type `Git: Clone` and press Enter.
    * Paste the repository URL: `[Sami: PASTE THE GITHUB REPO URL HERE]` and choose a location on your computer to save the project.
    * Once cloned, open the downloaded project folder in VS Code (`File -> Open Folder...`).

2.  **Open the Integrated Terminal in VS Code:**
    * Go to the menu `Terminal -> New Terminal` (or use the shortcut `Ctrl+`` `).
    * The terminal should open at the bottom, already inside the project folder.

3.  **Create Your Local Virtual Environment:**
    * In the VS Code terminal, type the following command and press Enter:
        ```bash
        python -m venv .venv
        ```
        *(Use `python3` if `python` doesn't point to Python 3 on your system. This creates a `.venv` folder just for this project on your machine. It's already listed in `.gitignore`.)*

4.  **Activate the Virtual Environment:**
    * **Crucial:** Activate the environment *in the same VS Code terminal*. The command depends on your operating system:
        * **Windows (using Command Prompt or PowerShell):**
            ```cmd
            .\.venv\Scripts\activate
            ```
        * **macOS / Linux (using bash or zsh):**
            ```bash
            source .venv/bin/activate
            ```
    * **Check:** You should see `(.venv)` appear at the start of your terminal prompt line inside VS Code.
    * *VS Code Hint:* VS Code might show a pop-up asking if you want to select the discovered environment `.venv` for the workspace. Click **Yes**.

5.  **Install Required Packages:**
    * Make sure the `(.venv)` prefix is visible in your terminal prompt. Then run:
        ```bash
        pip install -r requirements.txt
        ```
    * This reads the `requirements.txt` file from the repo and installs the exact versions of `pandas`, `scikit-learn`, etc., that were used for setup.

6.  **You're Ready!**
    * You can now open `Data_Exploration.ipynb` to see the EDA, examine the `.py` files, and run Python code using the correct isolated environment directly within VS Code.

## Next Steps & Roles (Immediate Focus)

**DEADLINE: Tuesday, April 22nd - Let's coordinate closely!**

1.  **Data Preprocessing Pipeline Design & Implementation (Lead: Marwan & Shehab):**
    * **Goal:** Create the Scikit-learn pipeline to clean and prepare data for modeling, based on EDA findings.
    * **Tasks:** Implement within `data_preprocessor.py` (and potentially `custom_transforms.py`, `evaluation_metrics.py`). Use `Pipeline` and `ColumnTransformer`.
        * Impute missing `ca` and `thal` values (discuss method: e.g., `SimpleImputer` with `strategy='most_frequent'`).
        * Scale numerical features (`StandardScaler`).
        * Encode categorical features (`OneHotEncoder`, consider `handle_unknown='ignore'`).
    * **Paradigms:** Focus on **OOP** structure (Marwan) integrating **Functional** components for specific transforms/metrics (Shehab).

2.  **Report Writing & Documentation (Lead: Abdulrahman):**
    * **Goal:** Start compiling the final report document.
    * **Tasks:**
        * Begin writing the **Introduction** and **Problem Description**.
        * Use EDA summary (from Sami/notebook) for the **Data Exploration** section.
        * Start drafting the **Solution Design** section, outlining the planned Preprocessing steps and the roles of OOP/Functional paradigms (coordinate with Marwan/Shehab).
        * Refer to the course requirements for report structure.

3.  **EDA Documentation & Support (Lead: Sami):**
    * **Goal:** Ensure EDA is well-documented and accessible.
    * **Tasks:**
        * Write a clear summary of key EDA findings (as discussed) and share with Abdulrahman.
        * Clean up and add comments/markdown explanations to `Data_Exploration.ipynb`.
        * Be available to answer questions about the data for the team.

4.  **Collaboration & Code Sharing:**
    * **Push all code changes** to this GitHub repository frequently (`git add .`, `git commit -m "Your message"`, `git push`).
    * Use the group chat for quick questions and coordination.
    * Update the shared Google Doc (link below) with notes or progress.

## Shared Document (Google Doc)
* https://docs.google.com/document/d/1QoEksyKeqQFb6kIcIq8cY7TUNYNvZNC9gxcSPhTlcyo/edit?usp=sharing
