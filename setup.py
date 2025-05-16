#!/usr/bin/env python3
from setuptools import setup, find_packages
import os
import sys

# Add src to path so we can import the version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from open_responses_server.version import __version__

setup(
    name='open-responses-server',
    version=__version__,
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "fastapi",
        "uvicorn",
        "httpx",
        "pydantic",
        "python-dotenv",
    ],
    entry_points={
        'console_scripts': [
            'otc=open_responses_server.cli:main',
        ],
    },
    author='Ori Nachum',
    author_email='ori.nachum@gmail.com',
    description='CLI to manage the OpenAI Responses Server that bridges chat completions to responses API calls',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/teabranch/open-responses-server',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Environment :: Web Environment',
    ],
    python_requires='>=3.7',
)