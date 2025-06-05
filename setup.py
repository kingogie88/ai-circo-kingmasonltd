from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="responsible-ai-implementation",
    version="0.1.0",
    author="kingogie88",
    author_email="kingogie88@github.com",
    description="A comprehensive framework for implementing responsible, ethical, and safe AI systems",
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
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.9.0",
            "isort>=5.10.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "mkdocs>=1.4.0",
            "mkdocs-material>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rai-bias=responsible_ai.cli.bias:main",
            "rai-fairness=responsible_ai.cli.fairness:main",
            "rai-explain=responsible_ai.cli.explain:main",
            "rai-privacy=responsible_ai.cli.privacy:main",
            "rai-safety=responsible_ai.cli.safety:main",
            "rai-governance=responsible_ai.cli.governance:main",
        ],
    },
) 