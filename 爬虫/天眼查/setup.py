from distutils.core import setup
setup(
  name = 'tianyancha',         # How you named your package folder (MyLib)
  packages = ['tianyancha'],   # Chose the same as "name"
  version = '2.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A Battery-included Scraper API of Tianyancha, the best Chinese Business Database.',   # Give a short description about your library
  author = 'Qiao Zhang',                   # Type in your name
  author_email = 'qiao.zhang.qz@outlook.com',      # Type in your E-Mail
  url = 'https://github.com/qzcool/Tianyancha',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/qzcool/Tianyancha/archive/v2.1.tar.gz',    # I explain this later on
  keywords = ['scraper', 'tianyancha', 'china', 'business'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'pandas',
          'bs4',
          'selenium',
      ],
  classifiers=[
    'Development Status :: 5 - Production/Stable',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)