# tmp_coding_challange

## First Steps

### Setup the Project

Install Poetry

`curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python`

And force Poetry to install the virtual environment in the project folder

`poetry config virtualenvs.in-project true`

Creates virtual environment and installs all dependencies and the project itself,
navigate into the folder with the file poetry.lock, and excecute:

`poetry install`