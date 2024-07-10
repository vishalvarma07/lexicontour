from setuptools import find_packages, setup
setup(
    name='lexicontour',
    packages=find_packages(include=['lexicontour']),
    version='0.2.0',
    description='Extracts contours and text from given document. Draws bounding regions over the first page of the document.',
    author='Divija Kanumury, Vishal Varma Kovoru, Pravardhitha Myneni',
    license='MIT',
)