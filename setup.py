"""
Hungarian loss function setup script.
"""

import setuptools


def get_long_description():
    """Reads the long project description from the 'README.md' file."""
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()


setuptools.setup(
    name="hungarian-loss",
    author="Mykola Galushka",
    author_email="mm.galushka@gmail.com",
    description=(
        "Package for computing the mean squared error between "
        "`y_true` and `y_pred` objects with prior assignment "
        "using the Hungarian algorithm."
    ),
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/mmgalushka/hungarian-loss",
    project_urls={
        "Bug Tracker": "https://github.com/mmgalushka/hungarian-loss/issues",
    },
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where=".", exclude=["tests"]),
    install_requires=["tensorflow>=2.4.0"],
    python_requires=">=3.6",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
)
