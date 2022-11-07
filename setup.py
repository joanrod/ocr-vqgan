from setuptools import setup, find_packages

setup(
    name='taming-transformers-ocrvqgan',
    version='0.0.1',
    description='Taming Transformers for High-Resolution Image Synthesis + OCR Perceptual Loss',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
