<div align="center">
  <img src="./assets/galirage_logo.png" width="100%" alt="galirage_logo" />
</div>
<hr>

# Arrow-Guided VLM: Enhancing Flowchart Understanding via Arrow Direction Encoding

## 🔍 Project Overview

<img src="./assets/pipeline_figure_vlm_train_data.png" width=100%>

This repository contains the source code used for flowchart diagram detection in the research of [Arrow-Guided VLM: Enhancing Flowchart Understanding via Arrow Direction Encoding].

The project mainly consists of the following two components:

-   `notebooks/`: Contains Jupyter Notebooks used for experiments on deep learning models for flowchart diagram detection. Each notebook may have its own setup instructions and dependency lists (e.g., `requirements.txt` within the directory).
-   `src/`: Contains Python scripts for tasks related to Large Language Models (LLMs), such as parsers and evaluation tools.

## Environment Setup

### LLM-related Scripts (`src/`)

Python scripts in the `src/` directory use a standard Python virtual environment (`venv`).

**Prerequisites:**
-   Python 3.11 or later

**Setup Steps:**
1. **Install Rye (if not already installed):**

    If you have not installed Rye yet, please follow the instructions on the official website based on your operating system:

    👉 https://rye.astral.sh/guide/installation/

2. **Clone this repository and move to the project root:**

    ```bash
    git clone https://github.com/galirage/Arrow-Guided-VLM-Enhancing-Flowchart-Understanding-via-Arrow-Direction-Encoding.git
    cd Arrow-Guided-VLM-Enhancing-Flowchart-Understanding-via-Arrow-Direction-Encoding
    ```

3. **Sync dependencies (installs Python and packages):**

    ```bash
    rye sync
    ```

### Deep Learning Notebooks (`notebooks/`)

Follow the instructions in each Jupyter Notebook in the `notebooks/` directory to set up the required environment and dependencies for specific experiments. If a `notebooks/requirements.txt` file is provided, you can use `pip` within any environment (e.g., conda, venv) to install the dependencies.

## Development Tools

For the code in the `src/` directory, it is recommended to use `ruff` as a linter/formatter and `pytest` for testing.  

```bash
rye run pytest
rye run ruff check .
```

If you are using VSCode, installing the `charliermarsh.ruf` extension will enable automatic formatting and real-time lint checking when saving code.

## OCR, Detection -> LLM Execution Procedure

1. Place a directory named `images/` (or any appropriate name) in a suitable location and add the input images there.

2. At the same directory level, create a directory named `json/` and store the output results in COCO data format.

3. Create a `.env` file by copying and editing the provided `.env.exsample` file.  
   Place the resulting `.env` file in the `Arrow-Guided-VLM-Enhancing-Flowchart-Understanding-via-Arrow-Direction-Encoding` directory.

   Example:
   ```bash
   cp .env.exsample .env
   ```
   Then edit .env to set your actual credentials

4. Execute `src/arrow-guided-vlm/graph` as a module using the command below:
   ```bash
   cd Arrow-Guided-VLM-Enhancing-Flowchart-Understanding-via-Arrow-Direction-Encoding/
   rye run python -m src.arrow-guided-vlm.graph --process_name all_image --img_dir PATH/TO/FLOW-CHART-IMAGE-DIRECTORY --output_dir PATH/TO/OUTPUT/DIR
   ```
   Example:
   ```
   rye run python -m src.arrow-guided-vlm.graph --process_name all_image --img_dir images/ --output_dir output
   ```

## dataset
### detection dataset
The `detection_data/` directory contains data for training and testing the detection model in the form of coco data. Among them, `detection_data/train/` is the data used for training and evaluation, and `detection_data/test/` is the test data used to test the training model.

### pipeline dataset
The images in the `images/` directory are samples of the data used in the pipeline. The corresponding inference results of the detection model are stored in the `json/` directory.