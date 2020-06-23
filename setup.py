from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

setup(
      name="ophys_etl_pipelines",
      use_scm_version=True,
      description=("Pipelines and transforms for processing optical "
                   "physiology data."),
      author="Kat Schelonka, Isaak Willett, Dan Kapner, Nicholas Mei",
      author_email="nicholas.mei@alleninstitute.org",
      url="https://github.com/AllenInstitute/ophys_etl_pipelines",
      package_dir={"": "src"},
      packages=find_packages(where="src"),
      setup_requires=["setuptools_scm"],
      install_requires=required,
)
