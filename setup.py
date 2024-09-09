from setuptools import setup, find_packages

setup(
    name='EQnMIX',
    version='2.0.1',
    description='A package for seismic probabilistc location using Gaussian Mixture Models ',
    author='Roberto Ortega',
    author_email='ortega@cicese.mx',
    url='https://github.com/rortegaru/EQnMIX',
    packages=find_packages(include=['EQNEMIX', 'EQNEMIX.*', 'utils', 'utils.*']),
    install_requires=[
        'numpy',
        'pandas',
        'nllgrid',
        'seisbench',
        'pymc',
        'obspy',  # Añade otras dependencias necesarias
    ],
    entry_points={
        'console_scripts': [
            'eqnmix=EQNEMIX.main:main_function',  # Ajusta según tu función principal
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
