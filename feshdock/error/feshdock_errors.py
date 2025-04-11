"""Custom error classes"""


class FeshdockError(Exception):
    """feshdock exception base class"""

    def __init__(self, cause):
        self.cause = cause

    def __str__(self):
        representation = "[%s] %s" % (self.__class__.__name__, self.cause)
        return representation


class FeshdockWarning(FeshdockError):
    """Custom error class intented only for warnings to be notified, not to fail"""

    pass


class RandomNumberError(FeshdockError):
    """Custom RandomNumber exception"""

    pass


class GSOError(FeshdockError):
    """Custom GSO exception"""

    pass


class GSOParameteresError(GSOError):
    """Custom GSOParameteres exception"""

    pass


class GSOCoordinatesError(GSOError):
    """Custom error for CoordinatesFileReader class"""

    pass


class StructureError(FeshdockError):
    """General structure error"""

    pass


class BackboneError(StructureError):
    """General structure error"""

    pass


class SideChainError(StructureError):
    """General structure error"""

    pass


class ResidueNonStandardError(StructureError):
    """General structure error"""

    pass


class AtomError(StructureError):
    """Atom error exception"""

    pass


class MinimumVolumeEllipsoidError(StructureError):
    """MinimumVolumeEllipsoid exception"""

    pass


class PDBParsingError(FeshdockError):
    """PDB parser error"""

    pass


class PDBParsingWarning(FeshdockWarning):
    """PDB parser warning"""

    pass


class PotentialsParsingError(FeshdockError):
    """Reading potential file error"""

    pass


class ScoringFunctionError(FeshdockError):
    """Error in the scoring function drivers"""

    pass


class NotSupportedInScoringError(FeshdockError):
    """Error to be raised when an atom or residue type is
    not supported by the scoring function"""

    pass


class NormalModesCalculationError(FeshdockError):
    """Error in normal modes calculation"""

    pass


class SetupError(FeshdockError):
    """Error in setup"""

    pass


class SwarmNumError(FeshdockError):
    """Error in number of swarms"""

    pass


class MembraneSetupError(FeshdockError):
    """Error in membrane setup"""

    pass
