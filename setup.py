from setuptools import setup

setup(
    name="zeta-guard",
    version="0.0.1",
    py_modules=["guard"],
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
    ],
    license="AGPL-3.0-only",  
    description="AI Training Stability Guardian with ζ=0.707 invariant",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
)
