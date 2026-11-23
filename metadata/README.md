# metadata

Small text files that describe the data. They are committed.

`sources.tsv`: one row per data source of the demo (`run` = `demo`, downloaded by
`scripts/download_data.py`) and of the full paper run (`run` = `full`, downloaded
on the cluster by the job script in `used_by`).

| Column | Meaning |
|--------|---------|
| `file` | where the file or folder goes, relative to the repository root |
| `run` | `demo` or `full` |
| `url` | where it is downloaded from |
| `provider` | who publishes it, with the DOI or accession |
| `license` | its license or terms of use |
| `provider_checksum` | the checksum the provider publishes (`md5:...`), or the commit for code (`git:...`) |
| `bytes` | file size |
| `used_by` | scripts that read it |

`checksums.sha256`: SHA-256 of the demo's raw files, checked by the download
script. Check them by hand from the repository root with
`shasum -a 256 -c metadata/checksums.sha256` (or `sha256sum -c` on Linux).
