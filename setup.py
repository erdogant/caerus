import setuptools
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     install_requires=['numpy','pandas','tqdm','matplotlib'],
     python_requires='>=3',
     name='caerus',  
     version='0.1.0',
#     version=versioneer.get_version(),    # VERSION CONTROL
#     cmdclass=versioneer.get_cmdclass(),  # VERSION CONTROL
     author="Erdogan Taskesen",
     author_email="erdogant@gmail.com",
     description="Detection of favorable moments",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/erdoganta/caerus",
	 download_url = 'https://github.com/erdoganta/caerus/archive/0.1.0.tar.gz',
     packages=setuptools.find_packages(), # Searches throughout all dirs for files to include
     include_package_data=True, # Must be true to include files depicted in MANIFEST.in
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: Apache Software License",
         "Operating System :: OS Independent",
     ],
 )
