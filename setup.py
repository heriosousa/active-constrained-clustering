import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="active-constrained-clustering",
    version="0.0.2",
    author="Herio Sousa",
    author_email="heriosousa@hotmail.com",
    description="Active constrained clustering algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/heriosousa/active-constrained-clustering",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'metric-learn>=0.4',
    ]
)
