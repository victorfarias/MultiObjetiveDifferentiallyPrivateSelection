# cython: infer_types=False, language_level=3, cdivision=True, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language = c++

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

# setup(ext_modules = cythonize('./mechanisms/generic/local_dampening_cy.pyx'))

ext_modules = [
    Extension(
        "local_sensitivity.metric",
        ["./local_sensitivity/metric.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
        # extra_link_args=['-fopenmp'],
        language="c++",
    ),
    Extension(
        "dp_mechanisms.delta_base",
        ["./dp_mechanisms/delta_base.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
        # extra_link_args=['-fopenmp'],
        language="c++",
    ),
    Extension(
        "dp_mechanisms.local_dampening_cy",
        ["./dp_mechanisms/local_dampening_cy.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
        # extra_link_args=['-fopenmp'],
        language="c++",
    ),    
    Extension(
        "pareto.pareto_score",
        ["./pareto/pareto_score.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
        # extra_link_args=['-fopenmp'],
        language="c++",
    ),
    Extension(
        "pareto.pareto_delta",
        ["./pareto/pareto_delta.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
        # extra_link_args=['-fopenmp'],
        language="c++",
    ),
    Extension(
        "tree.delta_tnr",
        ["./tree/delta_tnr.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
        # extra_link_args=['-fopenmp'],
        language="c++",
    ),    
    Extension(
        "tree.delta_tpr",
        ["./tree/delta_tpr.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
        # extra_link_args=['-fopenmp'],
        language="c++",
    ),     
    Extension(
        "dp_mechanisms.delta_weighted",
        ["./dp_mechanisms/delta_weighted.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
        # extra_link_args=['-fopenmp'],
        language="c++",
    ),    
    Extension(
        "local_sensitivity.egocentric_density",
        ["./local_sensitivity/egocentric_density.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
        # extra_link_args=['-fopenmp'],
        language="c++",
    ),
    Extension(
        "local_sensitivity.degree",
        ["./local_sensitivity/degree.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
        # extra_link_args=['-fopenmp'],
        language="c++",
    ),
    # Extension(
    #     "local_sensitivity.egocentric_betweenness_centrality",
    #     ["./local_sensitivity/egocentric_betweenness_centrality.pyx"],
    #     include_dirs=[numpy.get_include()],
    #     extra_compile_args=['-O3'],
    #     # extra_link_args=['-fopenmp'],
    #     language="c++",
    # ),
    #     Extension(
    #     "local_sensitivity.edges_alters",
    #     ["./local_sensitivity/edges_alters.pyx"],
    #     include_dirs=[numpy.get_include()],
    #     extra_compile_args=['-O3'],
    #     # extra_link_args=['-fopenmp'],
    #     language="c++",
    # ),
    #     Extension(
    #     "utils.utils_cy",
    #     ["./utils/utils_cy.pyx"],
    #     include_dirs=[numpy.get_include()],
    #     extra_compile_args=['-O3'],
    #     # extra_link_args=['-fopenmp'],
    #     language="c++",
    # ),
]

setup(
    name='local_dampening',
    ext_modules=cythonize(ext_modules),
)

# python setup.py build_ext --inplace
