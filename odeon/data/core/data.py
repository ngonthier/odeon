from typing import Type, TypeVar

# from odeon.core.exceptions import MisconfigurationException
from odeon.core.registry import GenericRegistry
from odeon.core.types import OdnData

T = TypeVar('T', bound='OdnData')

@GenericRegistry.register('data', aliases=['data_module'])
class DataRegistry(GenericRegistry[Type[T]]):
    @classmethod
    def register_fn(cls, cl: Type[OdnData], name: str):

        cls._registry[name] = cl
