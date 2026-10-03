"""Run with python -m embedit.cli --help."""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path

from .data import prompts_for, read_edits
from .pipelines import MODEL_IDS, REFINER_ID


def parser():
    p = argparse.ArgumentParser(description="Embedding editing for Stable Diffusion 1.4 and SDXL")
    p.add_argument("--model", required=True, choices=MODEL_IDS)
    p.add_argument("--model-id", help="Hugging Face model ID or a local Diffusers directory")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="User-supplied old,new CSV")
    source.add_argument("--load-edit", type=Path, help="Previously saved edit directory")
    p.add_argument("--output", type=Path, required=True, help="New, non-existing output directory")
    p.add_argument("--mode", choices=["sequential", "independent", "joint"], default="sequential")
    p.add_argument("--mse-factor", type=float)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--begin", type=int, default=0)
    p.add_argument("--end", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-seeds", type=int, default=1)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--prompt", action="append", default=[], help="Repeat for multiple generation prompts")
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--refiner", action="store_true", help="Use the SDXL refiner")
    p.add_argument("--refiner-id", default=REFINER_ID)
    p.add_argument("--local-files-only", action="store_true")
    return p


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.iterations < 1 or args.steps < 1 or args.num_seeds < 1:
        p.error("iterations, steps and num-seeds must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error("learning-rate must be finite and positive")
    if not math.isfinite(args.guidance):
        p.error("guidance must be finite")
    if args.mse_factor is not None and (not math.isfinite(args.mse_factor) or args.mse_factor < 0):
        p.error("mse-factor must be finite and nonnegative")
    if args.refiner and args.model != "sdxl":
        p.error("--refiner requires --model sdxl")
    if args.mode == "joint" and args.model != "sd14":
        p.error("the original joint-edit variant is available for SD1.4 only")
    if args.load_edit and (not args.prompt or args.skip_generation):
        p.error("--load-edit requires --prompt and image generation")
    if args.output.exists():
        p.error("output directory already exists; choose a new path")
    rows = []
    metadata = None
    try:
        if args.csv:
            rows = read_edits(args.csv, args.begin, args.end)
        else:
            metadata = json.loads((args.load_edit / "edit.json").read_text(encoding="utf-8"))
            if metadata["model"] != args.model:
                p.error("saved edit and --model do not match")
            if args.model_id and metadata["model_id"] != args.model_id:
                p.error("--model-id must match the base model used by the saved edit")
    except (OSError, ValueError, KeyError) as error:
        p.error(str(error))
    model_id = args.model_id or (metadata["model_id"] if metadata else MODEL_IDS[args.model])
    factor = args.mse_factor
    if factor is None:
        factor = 0.2 if args.mode == "joint" else (0.35 if args.model == "sd14" else 0.3)

    import torch
    from .editing import edit_embeddings
    from .pipelines import encoders, generate, load_pipeline

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA is unavailable; run inside your allocated GPU job")
    torch.manual_seed(args.seed)
    pipe, refiner = load_pipeline(args.model, model_id, args.device,
                                 args.refiner_id if args.refiner else None,
                                 args.local_files_only)
    encoder_pairs = encoders(pipe, args.model)
    args.output.mkdir(parents=True, exist_ok=False)
    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "diffusers", "transformers"]}
    save_json(args.output / "run.json", {**vars(args), "model_id": model_id,
                                        "mse_factor": factor,
                                        "versions": versions})

    def render(directory, selected):
        directory.mkdir(parents=True, exist_ok=True)
        image_metadata = []
        for number, (label, prompt) in enumerate(selected):
            for offset in range(args.num_seeds):
                seed = args.seed + offset
                image = generate(pipe, refiner, prompt, seed, args.steps, args.guidance, args.device)
                filename = f"prompt_{number:04d}_seed_{seed}.png"
                image.save(directory / filename)
                image_metadata.append({"file": filename, "label": label, "prompt": prompt, "seed": seed})
        save_json(directory / "prompts.json", image_metadata)

    def save_edit(directory, stats):
        directory.mkdir(parents=True, exist_ok=True)
        for name, encoder, _ in encoder_pairs:
            encoder.save_pretrained(directory / name, safe_serialization=True)
        save_json(directory / "edit.json", {"model": args.model, "model_id": model_id,
                                           "mode": args.mode, "statistics": stats})

    explicit_prompts = [("custom", value) for value in args.prompt]
    if args.load_edit:
        for name, encoder, _ in encoder_pairs:
            saved = type(encoder).from_pretrained(args.load_edit / name, local_files_only=True)
            encoder.load_state_dict(saved.state_dict(), strict=True)
            del saved
        render(args.output / "images", explicit_prompts)
        return

    original = {}
    if args.mode == "independent":
        original = {name: encoder.get_input_embeddings().weight.detach().cpu().clone()
                    for name, encoder, _ in encoder_pairs}
    groups = [rows] if args.mode == "joint" else [[row] for row in rows]
    all_stats = []
    for index, group in enumerate(groups):
        if original:
            with torch.no_grad():
                for name, encoder, _ in encoder_pairs:
                    encoder.get_input_embeddings().weight.copy_(original[name])
        stats = {"row_index": args.begin + index, "encoders": {}}
        for name, encoder, tokenizer in encoder_pairs:
            stats["encoders"][name] = edit_embeddings(
                encoder, tokenizer, [(row["old"], row["new"]) for row in group],
                learning_rate=args.learning_rate, iterations=args.iterations,
                factor=factor,
                joint=args.mode == "joint")
        all_stats.append(stats)
        if args.mode == "independent":
            destination = args.output / f"edit_{args.begin + index:04d}"
            save_edit(destination, [stats])
            if not args.skip_generation:
                render(destination / "images", explicit_prompts or prompts_for(group[0]))
    if args.mode != "independent":
        save_edit(args.output / "edited", all_stats)
        if not args.skip_generation:
            prompts = explicit_prompts or [item for row in rows for item in prompts_for(row)]
            render(args.output / "images", prompts)
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
