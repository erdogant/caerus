import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     install_requires=['numpy','pandas'],
     name='caerus',  
     version='0.1',
     author="Erdogan Taskesen",
     author_email="erdogant@gmail.com",
     description="Detection of favorable moments",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/erdoganta/CAERUS",
     packages=setuptools.find_packages(),
	 include_package_data=True,
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: Apache 2.0 License",
         "Operating System :: OS Independent",
     ],
 )
