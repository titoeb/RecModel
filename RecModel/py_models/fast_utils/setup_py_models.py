from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [Extension(name="sparse_tools", sources=["sparse_tools.pyx"], include_dirs=[np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    Extension(name="ease_utils", sources=["ease_utils.pyx"], include_dirs=[np.get_include()]),
    Extension(name="neighborhood_utils", sources=["neighborhood_utils.pyx"], include_dirs=[np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    Extension(name="slim_utils", sources=["slim_utils.pyx"], include_dirs=[np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'])]

setup(name="fast_utils", include_dirs = [np.get_include()], ext_modules=cythonize(extensions))

