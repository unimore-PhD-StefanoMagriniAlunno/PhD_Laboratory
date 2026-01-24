# Python Template
Repository template for python projects

## Get start
### Requirements
  - **Operative System Ubuntu 22.04 LTS**
    ```bash
    cat /etc/os-release
    ```
  - **Other installations:**
    ```bash
    git --version
    pdm --version
    ```
### Project Initialization
  Use this template to make a new repository, and clone it in your device.

  Initialize your project with `pdm`:
  ```bash
  pdm init
  ```
  follow the procedures, then:
  ``` bash
  pdm add pre-commit black mypy pytest
  pdm run pre-commit install
  ```
### Test
  Test your repository with your first commit:
  ```bash
  git add .
  git commit -m "Project initialization"
  git pull
  git push
  ```

### Data Science suite
  This is a list of packages for data science studies.
  
  Notebook suite:
  ```bash
  pdm add jupyter ipykernel notebook jupyterlab nbconvert
  ```
  
  Data management suite:
  ```bash
  pdm add pandas numpy
  ```

  Data visualisation suite:
  ```bash
  pdm add matplotlib seaborn plotly
  ```
  
  Scientific calculus suite:
  ```bash
  pdm add scipy scikit-learn statsmodels
  ```

  GPU boost suite:
  ```bash
  pdm add torch tensorflow
  ```
