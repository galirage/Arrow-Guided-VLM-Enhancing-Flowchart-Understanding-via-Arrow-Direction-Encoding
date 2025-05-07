<div align="center">
  <img src="./assets/galirage_logo.png" width="100%" alt="galirage_logo" />
</div>
<hr>

# Arrow-Guided VLM: Enhancing Flowchart Understanding via Arrow Direction Encoding

## 🔍 Project Overview

<img src="./assets/pipeline_figure_vlm_train_data.png" width=100%>

This repository contains the source code used for flowchart diagram detection in the research of [Paper Title (add if needed)].

The project mainly consists of the following two components:

-   `notebooks/`: Contains Jupyter Notebooks used for experiments on deep learning models for flowchart diagram detection. Each notebook may have its own setup instructions and dependency lists (e.g., `requirements.txt` within the directory).
-   `src/`: Contains Python scripts for tasks related to Large Language Models (LLMs), such as parsers and evaluation tools.

## Environment Setup

### LLM-related Scripts (`src/`)

Python scripts in the `src/` directory use a standard Python virtual environment (`venv`).

**Prerequisites:**
-   Python 3.11 or later

**Setup Steps:**

1.  Move to the root directory of the project:
    ```bash
    cd gg-rq-rag-flowchat-detection
    ```

2.  Create a virtual environment:
    ```bash
    python -m venv .venv
    ```

3.  Activate the virtual environment:
    -   On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```
    -   On Windows:
        ```bash
        .venv\Scripts\activate
        ```

4.  Install dependencies for the `src` directory (if `src/requirements.txt` exists):
    ```bash
    pip install -r src/requirements.txt
    ```
    *(Note: Create this `requirements.txt` file as needed.)*

### Deep Learning Notebooks (`notebooks/`)

Follow the instructions in each Jupyter Notebook in the `notebooks/` directory to set up the required environment and dependencies for specific experiments. If a `notebooks/requirements.txt` file is provided, you can use `pip` within any environment (e.g., conda, venv) to install the dependencies.

## Development Tools

For the code in the `src/` directory, it is recommended to use `ruff` as a linter/formatter and `pytest` for testing. Install them within the `.venv` environment you created:

```bash
pip install ruff pytest
```

## OCR, Detection -> LLM Execution Procedure
```bash
cd src
python arrow-guided-vlm/graph.py --process_name all_image --img_dir PATH/TO/IMAGES/DIR
# Example:
python arrow-guided-vlm/graph.py --process_name all_image --img_dir ../images/
```