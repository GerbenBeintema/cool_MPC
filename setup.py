from setuptools import setup, find_namespace_packages

with open('requirements.txt') as f:
    install_requires = [line for line in f]

name = 'cool_MPC'
packages = [a for a in find_namespace_packages(where='.') if a[:len(name)]==name]
print(packages)
setup(name = name,
      version = '0.1.0',
      description = 'An easy going MPC solver for torch f h system',
      author = 'Gerben Beintema',
      author_email = 'g.i.beintema@tue.nl',
      license = 'BSD 3-Clause License',
      python_requires = '>=3.6',
      packages=packages,
      install_requires = install_requires,
    )