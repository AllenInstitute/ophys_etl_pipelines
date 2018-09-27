from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
     requirements = f.read().splitlines()
 
with open('test_requirements.txt','r') as f:
     test_requirements = f.read().splitlines()

setup(
    name = 'mesoscope_2p',
    version = '0.1.3',
    description = """Tools for reading 2-photon mesoscope data""",
    author = "Jed Perkins",
    author_email = "jedp@alleninstitute.org",
    url = 'https://github.com/AllenInstitute/mesoscope_2p',
    packages = find_packages(),
    install_requires = requirements,
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'extract_planes = mesoscope_2p.scripts.extract_planes:main'
        ]
    },
    setup_requires=['pytest-runner'],
    tests_require = test_requirements
)
