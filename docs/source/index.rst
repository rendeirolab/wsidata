.. wsidata documentation master file, created by
   sphinx-quickstart on Sat Sep 14 12:10:32 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

wsidata: Efficient data structures and IO for whole slide image analysis
========================================================================


.. grid:: 1 2 2 2

   .. grid-item::
       :columns: 12 4 4 4

       .. image:: _static/logo.svg
          :align: center
          :width: 150px

   .. grid-item::
      :columns: 12 8 8 8
      :child-align: center

      `WSIData <wsidata.WSIData>`_ is data structure for manipulating whole slide image (WSI)
      and storing its analysis results in Python.
      It can read a variety of WSI formats and the storage is backed by SpatialData storage.
      :code:`wsidata` is designed to used with `LazySlide`.



.. toctree::
   :maxdepth: 3
   :hidden:

   installation
   intro/index
   api/index


.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      How to install wsidata

   .. grid-item-card:: Tutorial
      :link: intro/index
      :link-type: doc

      Introduction of Whole Slide Images and WSIData

   .. grid-item-card:: API
      :link: api/index
      :link-type: doc

      API Reference
