from setuptools import setup, find_packages

setup(
    name="jigsaw-toxic-comment-classification",
    version="0.1.0",
    author="Your Name",
    author_email="kausikdevanathan@gmail.com",
    description="A project for training a BERT model on the Jigsaw comment classification dataset.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "transformers",
        "scikit-learn",
        "pandas",
        "numpy",
        "tqdm",
        "matplotlib",
        "seaborn",
        "jupyter"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)