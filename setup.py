import os
from glob import glob # Indispensable pour trouver tous les fichiers launch
from setuptools import find_packages, setup

package_name = 'rl_tiago_dynamic'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Meriem L. AARIZOU',
    maintainer_email='meriem.aarizou@univ-mosta.dz',
    description='Packge to test RL algorithms on TIAGO Robot, in a dynamic environment',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    
    entry_points={
        'console_scripts': [
            'dynamic_obstacle_mover = rl_tiago_dynamic.dynamic_obstacle_mover:main'
        ],
    },
)
