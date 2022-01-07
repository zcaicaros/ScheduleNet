from setuptools import setup, find_packages

setup(
    name='pyjssp',
    version='0.5.0',
    description='A native python Job Shop Scheduling Problem (JSSP) simulator',
    author='Junyoung Park, Jaehyeong Chun',
    author_email='Junyoungpark@kaist.ac.kr',
    url='https://github.com/Junyoungpark/JSSPsimulator',
    install_requires=['numpy', 'matplotlib', 'plotly', 'networkx'],
    packages=find_packages(exclude=['tests']),
    keywords=['jssp', 'jssp simulator'],
    python_rquires='>=3',
    package_data={
        'benchmarks': [
            'benchmarks/*'
        ]
    },
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ]
)
