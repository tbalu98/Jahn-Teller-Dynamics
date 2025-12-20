#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        # Try different encodings (UTF-8, UTF-16 LE, UTF-16 BE)
        encodings = ['utf-8', 'utf-16-le', 'utf-16-be']
        for encoding in encodings:
            try:
                with open(readme_path, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        # If all encodings fail, return empty string
        return ""
    return ""

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name='jahn_teller_dynamics',
    version='1.0.2',  # Update this as needed
    description='Dynamic Jahn-Teller Effect Calculator',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='Balázs Tóth',
    author_email='toth.balazs@wigner.hun-ren.hu',
    url='https://github.com/tbalu98/Jahn-Teller-Dynamics',
    license='GPLv3',
    python_requires='>=3.10',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'Exe=jahn_teller_dynamics.Exe:main',
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)

