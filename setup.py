from distutils.core import setup
setup(
  name = 'cuprates_transport',         # How you named your package folder (MyLib)
  packages = ['cuprates_transport'],   # Chose the same as "name"
  version = '0.6',      # Start with a small number and increase it with every change you make
  license='gpl-3.0',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Computes Boltzmann transport from a tight binding model',   # Give a short description about your library
  author = 'Gael Grissonnanche',                   # Type in your name
  author_email = 'gael.phys@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/gaelgrissonnanche/cuprates_transport',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/gaelgrissonnanche/cuprates_transport/archive/v_06.tar.gz',    # I explain this later on
  keywords = ['Fermi Surface', 'Boltzmann', 'Solid State', 'Condensed Matter', 'Resistivity', 'Transport', 'Cuprates', 'Superconductivity'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'scipy',
          'matplotlib',
          'numba',
          'lmfit',
          'tqdm',
          'scikit-image',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',   # Again, pick a license
    'Programming Language :: Python :: 3.7',      #Specify which pyhton versions that you want to support
  ],
)