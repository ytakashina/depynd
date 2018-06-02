from setuptools import setup, find_packages

long_description = """`depynd` is a Python library for learning dependencies between random variables from data.
 It supports
- Mutual information (MI) estimation for discrete-continuous mixtures
- Feature selection algorithms based on MI
- Markov random field structure learning algorithms
"""

setup(
    name='depynd',
    version='0.4.5',
    description='Evaluating dependencies among random variables.',
    long_description=long_description,
    author='Yuya Takashina',
    author_email='takashina2051@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy', 'scipy', 'scikit-learn',
    ],
    url='https://github.com/y-takashina/depynd',
)
