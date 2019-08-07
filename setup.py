from setuptools import find_packages, setup

setup(
    name="openmdao-bridge-matlab",
    version="0.0.0",
    author="Olle Vidner",
    author_email="olle@vidner.se",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.6, <4",
    install_requires=["openmdao"],
)
