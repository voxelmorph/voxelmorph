import setuptools

setuptools.setup(
    name='voxelmorph',
    version='1.0.0',
    description='Image Registration with Convolutional Networks',
    url='https://github.com/voxelmorph/voxelmorph',
    python_requires='>=3.4',
    install_requires=[
        'numpy',
        'scipy',
        'sklearn',
        'pprint',
        'six',
        'pyyaml',
        'nibabel',
        'matplotlib',
        'tqdm',
    ]
)
