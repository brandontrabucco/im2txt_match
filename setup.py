from setuptools import setup

setup(name='im2txt_match',
      version='0.1',
      description='An image captioning framework using tensorflow',
      url='http://github.com/brandontrabucco/im2txt_match',
      author='Brandon Trabucco',
      author_email='brandon@btrabucco.com',
      license='MIT',
      packages=['im2txt_match', 'im2txt_match.inference_utils', 'im2txt_match.ops'],
      zip_safe=False)
