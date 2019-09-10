from setuptools import setup

with open("requirements.txt") as req_file:
    requirements = req_file.read().splitlines()

setup(
    name="catheter_ukf",
    version="0.1.0",
    packages=["catheter_ukf"],
    install_requires=requirements,
)
