#from distutils.core import setup
from setuptools import setup

#version = '0.1' # 
#version = '0.2' # - change trail detection thresholds
version = '0.2.1' # - error checking on fetch_calibs and calwf3

version_str = """# 
__version__ = "{0}"\n""".format(version)

fp = open('reprocess_wfc3/version.py','w')
fp.write(version_str)
fp.close()

setup(name='reprocess_wfc3',
      version=version,
      description='Reprocessing scripts for WFC3/IR',
      author='Gabriel Brammer',
      author_email='gabriel.brammer@nbi.ku.edu',
      url='https://github.com/gbrammer/reprocess_wfc3/',
      packages=['reprocess_wfc3'],
      install_requires=[]
     )
