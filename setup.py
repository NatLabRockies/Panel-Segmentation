try:
    from setuptools import setup
except ImportError:
    raise RuntimeError('setuptools is required')

import versioneer

KEYWORDS = [
    'photovoltaic',
    'solar',
    'analytics',
    'analysis',
    'performance',
    'PV'
    'satellite'
]


INSTALL_REQUIRES = [
    'opencv-python-headless==4.6.0.66',
    'numpy>=1.18.1',
    'scikit_image>=0.16.2',
    'matplotlib>=3.1.3',
    'requests>=2.22.0',
    'tensorflow>=2.2.0',
    'pandas>=1.1.4',
    'Pillow>=8.0.1',
    'scikit_learn>=0.23.2',
    'h5py>=2.10.0',
    'detecto>=1.2.1',
    'torch>=1.9.0',
    'torchvision>=0.10.0',
    'mmdet==3.3.0',
    'mmengine==0.10.7',
    'kornia',
    'scipy',
    'rasterio',
    'geopandas',
    'beautifulsoup4==4.12',
    'dash==2.6.0',
    'laspy[laszip]==2.5.4',
    'open3d==0.18.0',
    'pyarrow==17.0.0',
    'pyproj==3.5.0',
    'boto3>=1.37.38',
    'cfgrib>=0.9.15.0',
    'folium>=0.18.0',
    'simplekml>=1.3.6',
    'xarray>=2023.1.0',
]

TESTS_REQUIRE = [
    'pytest>=5.3.5'
]

EXTRAS_REQUIRE = {
    'doc': [
        'sphinx==5.0',
        'jinja2<3.1',
        'pydata-sphinx-theme==0.14.0',
        'ipython',
        'nbsphinx',
        "sphinx_copybutton"
    ],
    'test': TESTS_REQUIRE,
    'optional': []
}


setup(
    name='panel_segmentation',
    version=versioneer.get_version(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    extras_require=EXTRAS_REQUIRE,
    description='A package to segment solar panels from a '
                'satellite image and perform automated metadata extraction.',
    url='https://github.com/NREL/Panel-Segmentation',
    keywords=KEYWORDS,
    author='Kirsten Perry, Quyen Nguyen, Kevin Anderson, Christopher Campos, Ayobami Edun',
    author_email='kirsten.perry@nrel.gov; nguyquye@oregonstate.edu'
                'kevin.anderson@nrel.gov; chris.acampos@yahoo.com; aedun@ufl.edu',
    package_data={
        'panel_segmentation': [
            'panel_segmentation/examples/*',
            'panel_segmentation/tests/*',
            'panel_segmentation/models/VGG16Net_ConvTranpose_complete.h5',
            'panel_segmentation/models/VGG16_classification_model.h5',
            'panel_segmentation/models/object_detection_model.pth',
            'panel_segmentation/models/hail_config.py',
            'panel_segmentation/models/hail_model.pth',
            'panel_segmentation/models/post_hurricane_config.py',
            'panel_segmentation/models/post_hurricane_model.pth',
            'panel_segmentation/models/pre_hurricane_config.py',
            'panel_segmentation/models/pre_hurricane_model.pth',
            'panel_segmentation/models/sol_searcher_config.py',
            'panel_segmentation/models/sol_searcher_model.pth'],
    },
    include_package_data=True,
    license='MIT',
    packages=['panel_segmentation'],
    zip_safe=False
)
