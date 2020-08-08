import setuptools

setuptools.setup(
    name='voxelmorph',
    version='1.0.0',
    description='Image Registration with Convolutional Networks',
    url='https://github.com/voxelmorph/voxelmorph',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'numpy',
        'scipy',
        'nibabel',
    ]
)
