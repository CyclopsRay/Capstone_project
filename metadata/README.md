# metadata

- `sources.tsv`: where each raw file comes from: URL, provider, license, the provider's checksum, size, and the
  script that reads it
- `checksums.sha256`: SHA-256 of each raw file. `scripts/download_data.py` checks them; check by hand from the
  repository root with `shasum -a 256 -c metadata/checksums.sha256`
