from setuptools import setup, find_packages

__author__ = 'Remy Cazabet'
__license__ = "BSD-2-Clause"
__email__ = "remy.cazabet@gmail.com"

setup(name='structify_net',
      license='BSD-Clause-2',
      description='Network Generation with controlled structure',
      long_description = 'Network Generation with controlled structure',
      long_description_content_type = "text/markdown",
      url='https://github.com/Yquetzal/structify_net',
      author='Remy Cazabet',
      author_email='remy.cazabet@gmail.com',
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 4 - Beta',

          # Indicate who your project is intended for
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',

          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: BSD License',
          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python',
          'Programming Language :: Python :: 3'
      ],
      keywords='network-science community-detection',
      install_requires=[],
      packages=find_packages(),
      include_package_data=True
      )
