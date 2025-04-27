from setuptools import setup, find_packages

setup(
    name="nlp-ai-model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch==2.0.1",
        "transformers==4.30.2",
        "datasets==2.12.0",
        "seqeval==1.2.2",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.0.2",
        "fsspec==2024.12.0",
        "huggingface-hub>=0.14.1"
    ],
    python_requires=">=3.8",
) 