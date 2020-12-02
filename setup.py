import codecs
import re
from os import path

from setuptools import find_packages, setup

THIS_DIRECTORY = path.abspath(path.dirname(__file__))


def read(*parts):
    """ Read the content of a file and return a string of its content

    Args:
        parts (list(str)): The parts of the path of the file

    Returns:
        str: The string ot the file
    """
    with codecs.open(path.join(THIS_DIRECTORY, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    """ Read the version stored in __version__ attribute of a given path of a file

    ..note:
        taken from https://packaging.python.org/guides/single-sourcing-package-version/)

    Args:
        file_paths (list(str)): The parts of the path of the file

    Returns:
        str: The string of the version of the project
    """

    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def get_requirements(file_name):
    """Strip a requirement files of all comments

    Args:
        file_name (string): File which contains requirements

    Returns:
        list: list of requirements
    """
    with open(path.join(THIS_DIRECTORY, f"{file_name}.txt"), "r") as file:
        reqs = []

        for req in file.readlines():
            if not req.startswith("#"):
                print(req)
                if req.startswith("git+"):
                    name = req.split("#")[-1].replace("egg=", "").strip()
                    req.replace("git+", "")
                    reqs.append(f"{name} @ {req}")
                else:
                    reqs.append(req)

        return reqs


INSTALL_REQUIRES = get_requirements("requirements")
TESTS_REQUIRES = get_requirements("requirements-tests")

EXTRA_REQUIRES = {"tests": TESTS_REQUIRES}


setup(
    name="gechebnet",
    version=find_version("gechebnet", "__init__.py"),
    description="Description",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRA_REQUIRES,
    packages=find_packages(),
    include_package_data=True,
)
