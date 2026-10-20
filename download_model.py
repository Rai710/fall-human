"""Download the three VideoMAE inference files, without starting the app."""
import argparse
from pathlib import Path

MODEL_ID = "yadvender12/videomae-base-finetuned-kinetics-finetuned-fall-detect"
FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "models" / "videomae-fall"


def download_model(output=DEFAULT_OUTPUT, revision="main"):
    from huggingface_hub import snapshot_download

    output = Path(output)
    snapshot_download(repo_id=MODEL_ID, revision=revision,
                      allow_patterns=list(FILES), local_dir=str(output))
    if not all((output / name).is_file() for name in FILES):
        raise RuntimeError("Unduhan model belum lengkap. Jalankan kembali untuk melanjutkan.")
    print(f"Model VideoMAE tersimpan di: {output.resolve()}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revision", default="main", help="Branch/tag atau hash revisi Hugging Face.")
    args = parser.parse_args()
    download_model(args.output, args.revision)
