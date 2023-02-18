import setuptools

setuptools.setup(
    name="contact_graspnet",
    version="0.0.1",
    author="Moritz Hesche",
    author_email="mo.hesche@gmail.com",
    # description="",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    # url="",
    classifiers=["Programming Language :: Python :: 3"],
    packages=setuptools.find_packages(),
    # we do not specify a python version here, since the venv is always created with the same python version
    # as the one used to create the venv which might lead to errors during install
    install_requires=[
        "torch",
        "torchvision",
        "torchtyping",
        "torchsummary",
        "tensorboardX",
        "torchtyping",
        "Pillow",
        "numpy",
        "nptyping",
        "matplotlib",
        "pyyaml",
    ],
    extras_require={"dev": ["black", "pylint", "jupyter"]},
    include_package_data=True,
    use_scm_version=True,
)
