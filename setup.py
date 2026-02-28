# Importing required functions from setuptools library
# setuptools is used to package and distribute Python projects
from setuptools import find_packages, setup

# List is imported for type hinting (to specify return type clearly)
from typing import List


# This is a constant variable
# "-e ." is used in requirements.txt for editable installation
# We remove it later because setup() does not need it
HYPEN_E_DOT = '-e .'


# Function definition
# file_path: str → means input parameter should be string
# -> List[str] → means this function returns a list of strings
def get_requirements(file_path: str) -> List[str]:
    '''
    This function reads the requirements.txt file
    and returns the list of required packages
    '''

    # Empty list to store package names
    requirements = []

    # Open the requirements.txt file
    # with statement ensures file closes automatically
    with open(file_path) as file_obj:

        # Read all lines from file
        # Example: ["pandas\n", "numpy\n", "-e .\n"]
        requirements = file_obj.readlines()

        # Remove newline character "\n" from each line
        # So it becomes ["pandas", "numpy", "-e ."]
        requirements = [req.replace("\n", "") for req in requirements]

        # If "-e ." exists in list, remove it
        # Because setup() does not need this
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    # Return final cleaned list of packages
    return requirements


# setup() is the main function used to define project details
setup(

    # Name of your project (this becomes package name)
    name='mlproject',

    # Version of your project
    version='0.0.1',

    # Author name
    author='Sukriti',

    # Author email
    author_email='sukritiwaani2006@gmail.com',

    # Automatically finds all folders with __init__.py
    # So you don't have to manually list packages
    packages=find_packages(),

    # This installs all dependencies listed in requirements.txt
    # Example: pandas, numpy, seaborn
    install_requires=get_requirements('requirements.txt')
)