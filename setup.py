from setuptools import setup

setup(
    name="zeta-guard",
    version="0.0.1",
    py_modules=["guard"],  
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
    ],
)
