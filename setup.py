from distutils.core import setup

setup(name='depynd',
      version='0.2.2',
      description='Evaluating dependencies among random variables.',
      author='Yuya Takashina',
      author_email='takashina2051@gmail.com',
      packages=['depynd'],
      install_requires=[
          'numpy', 'scipy', 'sklearn',
      ],
      url='https://github.com/y-takashina/depynd',
      )
