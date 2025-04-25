#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name='openai-to-codex-wrapper',
    version='0.1.0',
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
            'otc=openai_to_codex_wrapper.cli:main',
        ],
    },
    author='Ori Nachum',
    author_email='ori.nachum@gmail.com',
    description='CLI to manage the OpenAI to Codex Wrapper API server',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/teabranch/openai-to-codex-wrapper',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Environment :: Web Environment',
    ],
    python_requires='>=3.7',
)