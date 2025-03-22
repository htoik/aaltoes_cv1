import torch
import torch.nn as nn
import torch.optim as optim
from torchdata.stateful_dataloader import StatefulDataLoader

from training_session import TrainingSession

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(embed_dim, 2, kernel_size=(1,1), padding=0),
            nn.Upsample(scale_factor=14, mode='nearest')
        )

    def forward(self, x):
        y = self.classifier(x).squeeze(2)
        return y

if __name__ == '__main__':
    import multiprocessing as mp
    import argparse
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser("train_noreuse.py")
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    workers = args.workers

    from datasets import get_config, get_dataset_noreuse, get_dataset_dashcam
    config = get_config()
    path_to_checkpoints = config['train']['path_to_checkpoints']
    session_name = config['train']['session_name']
    device = config['train']['device']
    batch_size = config["train"]["batch_size"]
    dino_model = config["train"]["models"]["dino_model"]
    input_dimension = config["train"]["input_dimension"]
    dataloader_workers = config["train"]["dataloader_workers"]
    save_every_n_batches = config['train']['save_every_n_batches']
    max_epochs = config['train']['max_epochs']

    session_name = f"{session_name}_noreuse"

    train_loader = StatefulDataLoader(
        get_dataset_noreuse(split='train'),
        batch_size=batch_size, num_workers=dataloader_workers, shuffle=True,
        collate_fn=collate_fn_safe)
    validation_loader = StatefulDataLoader(
        get_dataset_dashcam(split='validation'),
        # get_dataset_noreuse(split='validation'),
        batch_size=batch_size, num_workers=dataloader_workers, shuffle=False,
        collate_fn=collate_fn_safe)
    
    session = TrainingSession(
        train_loader, validation_loader,
        session_name=session_name,
        checkpoints_path=path_to_checkpoints, device=device
    )

    # for model_type in [ClassificationHead]:
    for model_type in [ClassificationHead, ClassificationHead2, ClassificationHead3, ClassificationHeadBilinear, SegmentationDecoder, SegmentationDecoder2, SegmentationDecoder3]:
        if sum(int(type(model) == model_type) for model, _, _ in session.model_session_generator()):
            print(f"Not creating model session for {model_type.__name__}. Already exists.")
        else:
            print(f"Creating model session for {model_type.__name__}.")
            model_session = session._create_model_session(model_type)
            session.add_model_session(model_session)

    dinov2 = torch.hub.load('facebookresearch/dinov2', dino_model).to(device)
    for param in dinov2.parameters():
        param.requires_grad = False
    dinov2.eval()
    
    criterion = nn.BCEWithLogitsLoss()

    from train_utils import train_one_epoch_with_session, calculate_validation_with_session
    for epoch in range(session.epoch, max_epochs):
        print(f" ---------  Starting epoch [{epoch+1: 2d}/{max_epochs: 2d}] ---------")
        for _ in train_one_epoch_with_session(session, dinov2, input_dimension, criterion, yield_interval=save_every_n_batches):
            calculate_validation_with_session(session, dinov2, input_dimension, criterion)
            session.save_checkpoint()
        calculate_validation_with_session(session, dinov2, input_dimension, criterion)
        session.epoch += 1
        session.save_checkpoint()
        print(f" ---------   Epoch done [{epoch+1: 2d}/{max_epochs: 2d}]    ---------")
