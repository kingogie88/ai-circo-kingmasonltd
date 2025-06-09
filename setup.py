from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="responsible-ai-implementation",
    version="0.1.0",
    author="KingMason Ltd",
    author_email="contact@kingmasonltd.com",
    description="A comprehensive framework for implementing responsible AI systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kingogie88/responsible-ai-implementation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "shap>=0.41.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "torch>=1.10.0",
        "tensorflow>=2.8.0",
        "aif360>=0.4.0",
        "diffprivlib>=0.5.0",
        "alibi>=0.8.0",
        "mlflow>=2.0.0",
        "great-expectations>=0.15.0",
        "safety-gym>=0.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.900",
            "sphinx>=4.5.0",
            "pre-commit>=2.17.0",
        ],
    },
) 