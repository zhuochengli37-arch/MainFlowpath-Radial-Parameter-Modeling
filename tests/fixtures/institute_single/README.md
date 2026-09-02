# Institute single-section fixtures

These files are the smallest representative subset of
`04_TEST_DATA/DATABASE-一个截面` required by automated tests. Numeric rows and
headers are preserved; repository line endings follow `.gitattributes`.

Included coverage:

- one `station=MAIN` file for FAN, CMP, HPTB and LPTB;
- one CMP INLET and one CMP OUTLET file for station recognition and exclusion;
- `CNC_*` speed parameters, component-specific `WnCOR_*` flow parameters, and
  the formal `xi Cpt Ctt Cps Cts MA` header.

No four-section file is copied here because four-section training and prediction
are outside the current phase.
