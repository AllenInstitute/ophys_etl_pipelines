from setuptools import setup, find_packages

setup(
    name = 'mesoscope_2p',
    version = '0.1.0',
    description = """Tools for reading 2-photon mesoscope data""",
    author = "Jed Perkins",
    author_email = "jedp@alleninstitute.org",
    url = 'https://github.com/AllenInstitute/mesoscope_2p',
    packages = find_packages(),
    include_package_data=True,
    setup_requires=['pytest-runner'],
)
