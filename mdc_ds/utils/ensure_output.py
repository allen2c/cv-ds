from pathlib import Path


def ensure_output(
    output: Path | str | None = None,
    *,
    output_dir: Path | str = ".",
    filename: str = "downloaded_dataset",
) -> Path:
    if output is None:
        output_dir = Path(output_dir)
        output = output_dir.joinpath(filename)
    else:
        output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
