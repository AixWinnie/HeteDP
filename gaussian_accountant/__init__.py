from .accountant import IAccountant
from .gdp import GaussianAccountant


__all__ = [
    "IAccountant",
    "GaussianAccountant"
]

def create_accountant(mechanism: str) -> IAccountant:
    if mechanism == "gdp":
        return GaussianAccountant()

    raise ValueError(f"Unexpected accounting mechanism: {mechanism}")
