"""

Organized datasets for training and prediction.

.. Note::
   Databases constructed from disk (i.e. anything besides the base ``Database`` class)
   will load floating point data in the format (float32 or float64)
   specified via the ``torch.get_default_dtype()`` function. Use ``torch.set_default_dtype()``
   to control this behavior.

"""
from .database import Database
from .ondisk import DirectoryDatabase, NPZDatabase

__all__ = ["Database", "DirectoryDatabase", "NPZDatabase"]