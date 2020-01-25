from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [Extension(name="sparse_tools", sources=["sparse_tools.pyx"], include_dirs=[np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    Extension(name="cd_fast", sources=["cd_fast.pyx"], include_dirs=[np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']), 
    Extension(name="base_class", sources=["base_class.pyx"], include_dirs=[np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    Extension(name="slim_model", sources=["slim_model.pyx"], include_dirs=[np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    Extension(name="naive_baseline_model", sources=["naive_baseline_model.pyx"], include_dirs=[np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    Extension(name="smart_baseline_model", sources=["smart_baseline_model.pyx"], include_dirs=[np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    Extension(name="baseline_model", sources=["baseline_model.pyx"], include_dirs=[np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    Extension(name="neighborhood_model", sources=["neighborhood_model.pyx"], include_dirs=[np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    Extension(name="ease_model", sources=["ease_model.pyx"], include_dirs=[np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'])]

setup(name="reco_models",
    include_dirs = [np.get_include()],
    ext_modules=cythonize(extensions))

