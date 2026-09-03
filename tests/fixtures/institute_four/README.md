# Institute Four-Section fixtures

These fixtures are minimal extracts from the authoritative formal sample at
`04_TEST_DATA/DATABASE-四个截面`.  Each `.dat` file preserves the source header
and the first two source data rows without changing any value.  The directory
layout is also preserved so component, stage, station, speed parameter and flow
parameter continue to be parsed from real path semantics.

The four MAIN files cover FAN, CMP, HPTB and LPTB.  FAN boundary files cover the
compressor-family `Vz/Rho` header; HPTB boundary files cover the turbine-family
`Ma` header.  These fixtures are adapter tests only and are not surrogate-model
training or accuracy data.
