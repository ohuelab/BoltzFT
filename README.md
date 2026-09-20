# BoltzFT

BoltzFT provides code to fine-tune the Boltz-2 affinity heads with experimental activity data and evaluate virtual screening performance.

## Plot results

With [uv](https://docs.astral.sh/uv/) installed:

```bash
make -C figures figures
```

This plots the supplied result CSVs in `figures/out/`.

## Run the analysis

See [Usage](docs/reproduce.md) for installation and commands.

| Directory | Contents |
| --- | --- |
| `pipeline/` | Binary-label input preparation, training and cached scoring |
| `evaluation/` | Screening metrics, cascade analysis, feature extraction and CV comparators |
| `patches/` | Boltz2_affinity changes required by the pipeline |
| `data/splits/` | Training and evaluation compound IDs |
| `figures/` | Plotting scripts and result CSVs |

## License

See [LICENSE](LICENSE) and [third-party licenses](licenses/).
