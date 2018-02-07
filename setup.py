from distutils.core import setup

setup(name='depynd',
      version='0.2.3',
      description='Evaluating dependencies among random variables.',
      author='Yuya Takashina',
      author_email='takashina2051@gmail.com',
      packages=['depynd',
                'depynd.bayesian_networks',
                'depynd.feature_selection',
                'depynd.markov_networks',
                'depynd.mutual_information' ,],
      install_requires=[
          'numpy', 'scipy', 'sklearn',
      ],
      url='https://github.com/y-takashina/depynd',
      )
