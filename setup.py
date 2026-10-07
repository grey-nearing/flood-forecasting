# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from setuptools import setup

# read the description from the README.md
readme_file = Path(__file__).absolute().parent / 'README.md'
with readme_file.open('r') as fp:
    long_description = fp.read()

about = {}
with open('model/__about__.py', 'r') as fp:
    exec(fp.read(), about)

setup(
    name='openhydronet',
    version=about['__version__'],
    packages=[
        'model',
        'model.datasetzoo',
        'model.datautils',
        'model.utils',
        'model.modelzoo',
        'model.training',
        'model.evaluation',
        'multimet',
        'multimet.utils',
        'multimet.catchment_delineation',
        'multimet.static_extractor',
        'multimet.gridded_archive_builders',
        'multimet.timeseries_extractors',
        'multimet.weather_fetcher',
        'return_periods',
        'benchmarks',
        'benchmarks.tools',
    ],
    package_data={
        'return_periods': ['*.csv'],
    },
    url='https://openhydronet.readthedocs.io',
    project_urls={
        'Documentation': 'https://openhydronet.readthedocs.io',
        'Source': 'https://github.com/google-research/flood-forecasting',
    },
    author='Amit Markel, Frederik Kratzert, Grey Nearing, Martin Gauch, Omri Shefi',
    author_email='flood-forecasting-open-source@google.com',
    description='Library for training deep learning models with environmental focus',
    long_description=long_description,
    long_description_content_type='text/markdown',
    entry_points={
        'console_scripts': [
            'schedule-runs=model.run_scheduler:_main',
            'run=model.run:_main',
            'build-cpc-archive=multimet.gridded_archive_builders.build_cpc_archive:main',
            'build-imerg-archive=multimet.gridded_archive_builders.build_imerg_archive:main',
            'extract-multimet=multimet.timeseries_extractors.runner:main',
            'extract-multimet-dask=multimet.timeseries_extractors.dask_runner:main',
            'multimet-realtime=multimet.timeseries_extractors.realtime:main',
            'sync-weather-forecasts=multimet.weather_fetcher.cli:main',
            'delineate-catchment=multimet.catchment_delineation.cli:main',
            'extract-caravan-static=multimet.static_extractor.cli:main',
            'extract-static-attributes=multimet.static_extractor.cli:main',
            'extract-caravan-static-batch=multimet.static_extractor.batch_runner:main',
            'extract-static-attributes-batch=multimet.static_extractor.batch_runner:main',
            'benchmark-catchment=benchmarks.catchment_delineation:main',
            'benchmark-static-extractor=benchmarks.static_extractor:main',
            'benchmark-gridded-archive=benchmarks.gridded_archive_builders:main',
            'benchmark-timeseries-extractor=benchmarks.timeseries_extractors:main',
            'benchmark-return-periods=benchmarks.return_periods:main',
            'benchmark-model=benchmarks.model:main',
        ]
    },
    python_requires='>=3.12',
    install_requires=[
        'absl-py',
        'bokeh',
        'cachey',
        'dask',
        'fsspec',
        'gcsfs',
        'geopandas',
        'google-cloud-storage',
        'h5netcdf',
        'h5py',
        'matplotlib',
        'more-itertools',
        'netcdf4',
        'numpy',
        'pandas',
        'pyarrow',
        'pydantic',
        'pyogrio',
        'pyproj',
        'rasterio',
        'requests',
        'ruamel.yaml',
        'scipy',
        'seaborn',
        'shapely',
        'tensorboard',
        'torch',
        'tqdm',
        'xarray',
        'zarr',
    ],
    extras_require={
        'dynamical': [
            'dynamical-catalog',
            'icechunk',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Hydrology',
        'License :: OSI Approved :: Apache Software License',
    ],
    keywords='deep learning hydrology lstm neural network streamflow discharge rainfall-runoff',
)
