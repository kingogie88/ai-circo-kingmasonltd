"""Setup script for the plastic recycling system."""

from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read long description
with open('README.md') as f:
    long_description = f.read()

setup(
    name='ai-circo-plastic-recycling',
    version='0.1.0',
    description='AI-Powered Plastic Recycling System',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='KingMason Ltd',
    author_email='support@kingmasonltd.com',
    url='https://github.com/kingogie88/ai-circo-kingmasonltd',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.2.5',
            'pytest-cov>=2.12.0',
            'black>=21.7b0',
            'flake8>=3.9.0',
            'mypy>=0.910',
            'isort>=5.9.0',
        ],
        'docs': [
            'sphinx>=4.1.0',
            'sphinx-rtd-theme>=0.5.2',
            'nbsphinx>=0.8.7',
        ],
        'gpu': [
            'torch>=1.9.0',
            'torchvision>=0.10.0',
            'tensorflow-gpu>=2.6.0',
        ],
    },
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