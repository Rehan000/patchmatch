from setuptools import setup, find_packages

setup(
    name='patchmatch',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'opencv-python==4.11.0.86',
        'numpy==2.1.3',
        'tqdm==4.67.1',
        'tensorflow==2.19.0',
        'pyyaml==6.0.2',
    ],
    author='Muhammad Rehan',
    description='A lightweight patch-based image matching package using Siamese descriptors.',
    python_requires='>=3.8',
)
