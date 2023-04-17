from setuptools import setup

setup(name='image_score',
      version='0.1',
      description='Compare images containing people with generated images meant to resemble the original',
      url='https://github.com/niuniujiaojiao/stable-diffusion-490/tree/main/code/final/image_score',
      author='niuniujiaojiao',
      author_email='crystal.wang@yale.edu',
      license='',
      packages=['image_score'],
      install_requires=['numpy', 'PIL', 'scipy', 'dlib', 'face_recognition', 'itertools', 'brisque'],
      zip_safe=False)
