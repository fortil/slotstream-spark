# Not the upstream file (slotstream-spark): the upstream package __init__ imports every fla subpackage.
# Only the gated-delta-rule closure is vendored, so this module carries the version alone.
__version__ = "0.5.2"
__all__: list[str] = []
