from setuptools import setup

setup(name='image_score',
      version='0.1',
      description='Compare images containing people with generated images meant to resemble the original',
      url='https://github.com/niuniujiaojiao/image_score',
      author='niuniujiaojiao',
      author_email='crystal.wang@yale.edu',
      license='',
      packages=['image_score'],
      install_requires=['numpy', 'Pillow', 'scipy', 'dlib', 'face_recognition', 'brisque'],
      zip_safe=False)
