import os

import setuptools

about: dict = {}
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "pipelines", "__version__.py")) as f:
    exec(f.read(), about)

with open("README.md") if os.path.exists("README.md") else open(os.devnull) as f:
    long_description = f.read()

setuptools.setup(
    name=about["__title__"],
    description=about["__description__"],
    version=about["__version__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=["tests"]),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=["sagemaker>=3.5.0,<4.0", "boto3", "pyyaml", "pandas", "scikit-learn"],
    extras_require={"test": ["pytest", "pytest-cov"]},
    entry_points={
        "console_scripts": [
            "get-pipeline-definition=pipelines.get_pipeline_definition:main",
            "run-pipeline=pipelines.run_pipeline:main",
        ]
    },
)
