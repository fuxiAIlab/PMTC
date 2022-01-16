from setuptools import setup, find_packages

# Some lessons learned:
# 1. Reinstall by pip will only work if you change the version every time.
# 2. To include static files, you need MANIFEST.in and change below configs
# 3. You MUST put in "__init__.py" in each directory if you want to add them

setup(name='game_seq_embedder',
      version='0.0.9',
      description='description for game_seq_embedder',
      url='',
      author='pujiashu',
      author_email='iamlxb3@gmail.com',
      license='MIT',
      packages=find_packages(),
      package_data={'templates': ['*'], 'static': ['*'], 'docs': ['*'], },
      include_package_data=True,
      zip_safe=False)
