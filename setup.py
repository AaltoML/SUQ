from setuptools import setup, find_packages

setup(
    name='suq',
    version='0.1.0',
    description='Streamlined Uncertainty Quantification (SUQ)',
    author='Rui Li, Marcus Klasson, Arno Solin, Martin Trapp',
    url='https://github.com/AaltoML/SUQ',
    packages=find_packages(exclude=["examples*", "tests*"]),
    install_requires=[
        'torch>=1.10',
        'numpy>=1.21',
        'tqdm>=4.60'
    ],
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
