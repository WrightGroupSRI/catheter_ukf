from setuptools import setup

setup(
    name="catheter_ukf",
    version="0.1.0",
    packages=["catheter_ukf"],
    install_requires=[
        "numba",
        "numpy",
        "numpy-quaternion",
        "scipy",
    ],
)
