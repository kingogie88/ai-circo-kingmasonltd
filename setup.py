"""Setup script for the plastic recycling system."""

from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read long description
with open('README.md') as f:
    long_description = f.read()

setup(
    name="plastic-recycling-system",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.2",
        "pandas>=1.2.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "RPi.GPIO>=0.7.0",
        "loguru>=0.5.3",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "pytest-asyncio>=0.15.1",
            "pytest-mock>=3.6.1",
            "black>=21.5b2",
            "flake8>=3.9.2",
            "mypy>=0.910",
            "isort>=5.9.3",
            "types-setuptools>=57.4.0",
            "types-PyYAML>=6.0.0",
        ],
    },
    description="AI-powered plastic recycling system with safety monitoring",
    author="KingMason Ltd",
    author_email="info@kingmason.com",
    url="https://github.com/kingogie88/ai-circo-kingmasonltd",
    entry_points={
        'console_scripts': [
            'plastic-recycling=src.main:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
) 