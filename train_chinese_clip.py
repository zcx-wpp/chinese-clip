import argparse
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

from training_data import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal Chinese-CLIP fine-tuning script.")
    parser.add_argument(
        "--jsonl-path",
        default=r"C:\Users\24703\Desktop\chinese_clip\cleaned_muge_with_negatives.jsonl",
        help="Training JSONL path.",
    )
    parser.add_argument(
        "--model-path",
        default=r"C:\Users\24703\Desktop\chinese_clip\model",
        help="Local Chinese-CLIP model directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=r"C:\Users\24703\Desktop\chinese_clip\checkpoints",
        help="Directory for checkpoints.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--batch-strategy",
        choices=["random", "hard_negative"],
        default="random",
        help="How to organize samples within each batch.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional cap on optimization steps. 0 means full epoch(s).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=200,
        help="Save a checkpoint every N steps.",
    )
    parser.add_argument(
        "--train-mode",
        choices=["projection_only", "full"],
        default="projection_only",
        help="Training scope. projection_only freezes most parameters for lightweight tuning.",
    )
    return parser.parse_args()


def ensure_feature_tensor(features):
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "text_embeds") and features.text_embeds is not None:
        return features.text_embeds
    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        return features.image_embeds
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
        return features.last_hidden_state[:, 0]
    raise TypeError(f"Unsupported feature output type: {type(features)!r}")


def clip_contrastive_loss(model, batch, device):
    text_inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }
    if "token_type_ids" in batch:
        text_inputs["token_type_ids"] = batch["token_type_ids"].to(device)

    image_inputs = {
        "pixel_values": batch["pixel_values"].to(device),
    }

    text_features = model.get_text_features(**text_inputs)
    image_features = model.get_image_features(**image_inputs)

    text_features = ensure_feature_tensor(text_features)
    image_features = ensure_feature_tensor(image_features)

    text_features = F.normalize(text_features, dim=-1)
    image_features = F.normalize(image_features, dim=-1)

    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    labels = torch.arange(logits_per_image.size(0), device=device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i + loss_t) / 2

    return loss, logits_per_image.detach()


def save_checkpoint(model, processor, optimizer, epoch, step, output_dir):
    checkpoint_dir = Path(output_dir) / f"epoch_{epoch:02d}_step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoint to: {checkpoint_dir}")
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    torch.save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
        },
        checkpoint_dir / "trainer_state.pt",
    )
    required_files = [
        checkpoint_dir / "config.json",
        checkpoint_dir / "trainer_state.pt",
    ]
    missing = [str(path.name) for path in required_files if not path.exists()]
    weight_files = list(checkpoint_dir.glob("pytorch_model*.bin")) + list(
        checkpoint_dir.glob("model-*.safetensors")
    ) + list(checkpoint_dir.glob("pytorch_model*.safetensors")) + list(
        checkpoint_dir.glob("model.safetensors")
    )
    if not weight_files:
        missing.append("model weights")
    if missing:
        raise RuntimeError(f"Checkpoint save incomplete, missing files: {missing}")
    print(f"Checkpoint files: {[path.name for path in sorted(checkpoint_dir.iterdir())]}")
    print(f"Saved checkpoint to: {checkpoint_dir}")


def configure_trainable_parameters(model, train_mode):
    if train_mode == "full":
        for param in model.parameters():
            param.requires_grad = True
        return

    for param in model.parameters():
        param.requires_grad = False

    for param in model.visual_projection.parameters():
        param.requires_grad = True

    for param in model.text_projection.parameters():
        param.requires_grad = True


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Building dataloader...")
    dataloader = build_dataloader(
        jsonl_path=args.jsonl_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        batch_strategy=args.batch_strategy,
    )

    print("Loading model...")
    processor = ChineseCLIPProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = ChineseCLIPModel.from_pretrained(args.model_path, local_files_only=True).to(device)
    configure_trainable_parameters(model, args.train_mode)
    model.train()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    trainable_param_count = sum(param.numel() for param in trainable_params)
    print(f"Train mode: {args.train_mode}")
    print(f"Trainable parameter count: {trainable_param_count}")
    print(f"Batch strategy: {args.batch_strategy}")

    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        for batch in dataloader:
            global_step += 1
            try:
                print(f"[step {global_step}] batch loaded: batch_size={len(batch['ids'])}")
                optimizer.zero_grad(set_to_none=True)

                loss, logits_per_image = clip_contrastive_loss(model, batch, device)
                print(f"[step {global_step}] forward done")

                loss.backward()
                print(f"[step {global_step}] backward done")

                optimizer.step()
                print(f"[step {global_step}] optimizer step done")

                with torch.no_grad():
                    batch_acc = (
                        logits_per_image.argmax(dim=1)
                        == torch.arange(logits_per_image.size(0), device=device)
                    ).float().mean().item()

                print(
                    f"step={global_step} "
                    f"loss={loss.item():.4f} "
                    f"batch_t2i_acc={batch_acc:.4f}"
                )

                if args.save_every > 0 and global_step % args.save_every == 0:
                    save_checkpoint(model, processor, optimizer, epoch, global_step, output_dir)

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
            except Exception as exc:
                print(f"[step {global_step}] training failed: {exc}")
                traceback.print_exc()
                raise

        save_checkpoint(model, processor, optimizer, epoch, global_step, output_dir)
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    print("Training finished.")


if __name__ == "__main__":
    main()
