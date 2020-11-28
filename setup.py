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
        (list): list of requirements
        (list): list of dependecy links
    """
    with open(path.join(THIS_DIRECTORY, "{}.txt".format(file_name)), "r") as file:
        reqs = []
        dep_links = []

        for line in file.readlines():
            if not line.startswith("#"):
                if "-f" in line:
                    _, dep_link = line.split("-f")
                    dep_links.append(dep_link)

                else:
                    reqs.append(line)

        return reqs, dep_links


INSTALL_REQUIRES, DEPENDENCY_LINKS = get_requirements("requirements")
TESTS_REQUIRES, _ = get_requirements("requirements-tests")

EXTRA_REQUIRE = {"tests": TESTS_REQUIRES}


setup(
    name="gechebnet",
    version=find_version("gechebnet", "__init__.py"),
    description="Description",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    install_requires=INSTALL_REQUIRES,
    dependency_lins=DEPENDENCY_LINKS,
    extras_require=EXTRA_REQUIRE,
    packages=find_packages(),
    include_package_data=True,
)
