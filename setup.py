import setuptools

setuptools.setup(
    name='edstan',
    version='0.1',
    packages=['edstan'],
    license='BSD-3-Clause',
    description='Fits item response theory models using psytan',
    #long_description=open('README.txt').read(),
    install_requires=['pystan', 'numpy', 'patsy', 're', 'io', 'contextlib'],
    url='https://?',
    author='Daniel Furr',
    author_email='danielcfurr@gmail.com',
    package_dir = {'edstan': 'edstan'},
    data_files = [('models', ['models/*.stan'])]
)
