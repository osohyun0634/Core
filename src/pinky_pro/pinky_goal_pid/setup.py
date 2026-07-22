from setuptools import find_packages, setup

package_name = 'pinky_goal_pid'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pinky',
    maintainer_email='pinky@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'goal_pid = pinky_goal_pid.goal_pid:main',
        'goal_pd = pinky_goal_pid.goal_pd:main',
        'goal_pid_curve = pinky_goal_pid.goal_pid_curve:main',
        'rpt_pd = pinky_goal_pid.rpt_pd:main', 
    ],
},
)
