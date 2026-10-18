# Transform-feedback research

The transform-feedback series from `mr-44928-upstream-latest` is available with
`MESA_KK_EXPERIMENTAL=transform_feedback`. It is disabled by default. This mode
advertises transform feedback and geometry streams, and disables graphics
pipeline libraries and shader objects for that process. Do not set it globally
or in shipped runtime defaults.

The old series predates the fork's pipeline libraries, shader objects, point
emulation, and cache ownership changes. In particular, a pre-rasterization GPL
library need not provide input topology, so a passthrough capture shader cannot
be specialized there using the old implementation. Capturing with GPL or
shader objects remains unfinished. Dynamic topology without an input assembly
state is rejected by the research compiler rather than dereferenced.

The port retains existing point emulation and cache recovery, copies geometry
metadata and owned index storage into pipeline vertex shaders, and versions
serialized shader metadata and auxiliary programs. Production capability tests
require the existing GPL/shader-object features and no transform feedback;
separate research tests require the inverse. These tests query the actual
compiled driver on a physical device, and skip only when no device is exposed.

Local validation on macOS 27.2 / Apple A18 Pro: all eight KosmicKrisp native
suites pass, including both capability contracts. Before isolating the research
mode, CTS basic point capture, triangle capture, and resume passed; fast GPL
capture stalled during shader compilation. That stall is a remaining feature
integration limit, not a successful GPL test. A full conformance run, shader
object capture, the macOS 26 deployment lane, Windows/Rosetta rendering, and
gameplay acceptance are still required before default enablement.
