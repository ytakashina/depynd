from setuptools import setup, find_packages

setup(name='depynd',
      version='0.3.2',
      description='Evaluating dependencies among random variables.',
      author='Yuya Takashina',
      author_email='takashina2051@gmail.com',
      packages=find_packages(),
      install_requires=[
          'numpy', 'scipy', 'sklearn',
      ],
      url='https://github.com/y-takashina/depynd',
      )
