from distutils.core import setup
#from setuptools import setup

# v0.1
# v0.2 - change trail detection thresholds

setup(name='reprocess_wfc3',
      version='0.2',
      description='Reprocessing scripts for WFC3/IR',
      author='Gabriel Brammer',
      author_email='gabriel.brammer@nbi.ku.edu',
      url='https://github.com/gbrammer/reprocess_wfc3/',
      packages=['reprocess_wfc3'],
      install_requires=['numpy', 'matplotlib', 
                        'astropy', 'scikit-image', 'shapely']
     )
