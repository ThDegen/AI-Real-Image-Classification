import torch
import hydra
from omegaconf import DictConfig

from ai_real_image_classification.model import Model


@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = "./models/best.ckpt"

    model = Model.load_from_checkpoint(
        checkpoint_path,
        n_class=cfg.data.num_classes,
        pretrained=False,
        model_name="resnet18",
    )

    model.eval()
    model.to(device)

    dummy_input = torch.randn(1, 3, cfg.data.img_size, cfg.data.img_size, device=device)

    model.to_onnx(
        file_path="./models/resnet18.onnx",
        input_sample=dummy_input,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print("ONNX export completed")


if __name__ == "__main__":
    main()
