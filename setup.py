python
from setuptools import setup, find_packages

setup(
    name='image-captioning-project',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'pandas',
        'torch',
        'transformers',
        'torchtext',
        'Pillow',
        'matplotlib',
        'requests'
    ],
)
