# Official-source calibration corpus

This directory contains scholarship records assigned exclusively to the
ScholarAgent calibration partition.

The records may be used to select the dense-retrieval threshold and other
parameters explicitly allowed by `docs/evaluation_protocol.md`. They must not
be used as held-out final evaluation evidence.

Each record must:

- be supported by an official source snapshot under
  `data/sources/calibration/`;
- have a matching provenance entry in `source_manifest.json`;
- preserve the official URL, source-check date and SHA-256 hash;
- encode unresolved conditions as manual-review requirements;
- avoid requirements that are not supported by the saved official source.

The current corpus is incomplete and will contain six scholarship identities
before calibration begins.

Current progress: 6 of 6 planned calibration scholarships.

The scholarship corpus is complete, but the calibration benchmark and parameter-selection run have not yet been created.
