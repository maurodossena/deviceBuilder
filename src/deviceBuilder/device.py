"""The `Device` class: a Hamiltonian, an overlap and the lattice they live on."""

from .bands import BandStructureMixin
from .base import DeviceBase
from .export import ExportMixin
from .glue import GlueMixin
from .loaders import LoaderMixin
from .transform import TransformMixin
from .upscale import UpscaleMixin
from .visualize import VisualizationMixin


class Device(
    LoaderMixin,
    TransformMixin,
    GlueMixin,
    UpscaleMixin,
    BandStructureMixin,
    VisualizationMixin,
    ExportMixin,
    DeviceBase,
):
    """
    A device, i.e. a set of sparse Hamiltonian and overlap blocks together with
    the lattice they are defined on.

    The behaviour is split over one mixin per concern, all sharing the state
    defined in `DeviceBase`:

    * `LoaderMixin`        : reading a device from CP2K, Wannier90 or MALOQ output.
    * `TransformMixin`     : moving, sorting and trimming the lattice, shifting energies.
    * `GlueMixin`          : joining two devices along an interface.
    * `UpscaleMixin`       : repeating the device, or one of its contacts.
    * `BandStructureMixin` : diagonalizing H(k) along a k-path.
    * `VisualizationMixin` : viewing the lattice and on-site quantities.
    * `ExportMixin`        : writing the QUATREX and OMEN input decks.
    """
