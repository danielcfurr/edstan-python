from setuptools import setup, find_packages

setup(
    name='edstan',
    version='0.2.0',
    packages=find_packages(),
    description='Streamlimes the fitting of common Bayesian item response theory models using Stan',
    author='Daniel C. Furr',
    author_email='danielcfurr@berkeley.edu',
    url='https://github.com/danielcfurr/edstan-python',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

