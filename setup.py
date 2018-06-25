from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
     requirements = f.read().splitlines()
 
with open('test_requirements.txt','r') as f:
     test_requirements = f.read().splitlines()

setup(
    name = 'brain_observatory_utils',
    version = '0.1.0',
    description = """Utilities for brain observatory pipelines""",
    author = "Jed Perkins",
    author_email = "jedp@alleninstitute.org",
    url = 'https://github.com/AllenInstitute/brain_observatory_utils',
    packages = find_packages(),
    install_requires = requirements,
    include_package_data=True,
    setup_requires=['pytest-runner'],
    tests_require = test_requirements
)
