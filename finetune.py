import gc
import os
import torch
import torch.nn as nn
import numpy as np
import yaml
import shutil
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from losses.loss import CombinedLoss
from datasets.dataset import Chunked_sample_dataset, img_batch_tensor2numpy
from models.mem_cvae import HFVAD
from utils.initialization_utils import weights_init_kaiming
from utils.vis_utils import visualize_sequences
from utils.model_utils import loader, saver, only_model_saver
from eval import evaluate


def train(config, training_chunked_samples_dir, testing_chunked_samples_file):
    paths = dict(log_dir="%s/%s" % (config["log_root"], config["exp_name"]),
                 ckpt_dir="%s/%s" % (config["ckpt_root"], config["exp_name"]))
    os.makedirs(paths["ckpt_dir"], exist_ok=True)

    batch_size = config["batchsize"]
    epochs = config["num_epochs"]
    num_workers = config["num_workers"]
    device = config["device"]
    lr = config["lr"]
    training_chunk_samples_files = sorted(os.listdir(training_chunked_samples_dir))

    # Loss wrapper (CombinedLoss includes intensity/grad/percep/ssim)
    combined_loss = CombinedLoss(config, device=device)
    combined_loss = combined_loss.to(device)

    model = HFVAD(num_hist=config["model_paras"]["clip_hist"],
                  num_pred=config["model_paras"]["clip_pred"],
                  config=config,
                  features_root=config["model_paras"]["feature_root"],
                  num_slots=config["model_paras"]["num_slots"],
                  shrink_thres=config["model_paras"]["shrink_thres"],
                  mem_usage=config["model_paras"]["mem_usage"],
                  skip_ops=config["model_paras"]["skip_ops"],
                  finetune=config["model_paras"]["finetune"]).to(device)

    # optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-7, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    step = 0
    epoch_last = 0
    if not config.get("pretrained"):
        model.apply(weights_init_kaiming)
    else:
        model_state_dict, optimizer_state_dict, step = loader(config["pretrained"])
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)

    writer = SummaryWriter(paths["log_dir"])
    shutil.copyfile("./cfgs/finetune_cfg.yaml", os.path.join(config["log_root"], config["exp_name"], "finetune_cfg.yaml"))

    best_auc = -1

    use_amp = config.get("use_amp", True)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    grad_clip = config.get("grad_clip", 1.0)

    for epoch in range(epoch_last, epochs + epoch_last):
        for chunk_file_idx, chunk_file in enumerate(training_chunk_samples_files):
            dataset = Chunked_sample_dataset(os.path.join(training_chunked_samples_dir, chunk_file))
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            pbar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1} chunk {chunk_file_idx}", total=len(dataloader))
            for idx, train_data in pbar:
                model.train()
                sample_frames, sample_ofs, _, _, _ = train_data
                sample_frames = sample_frames.to(device)
                sample_ofs = sample_ofs.to(device)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=use_amp):
                    out = model(sample_frames, sample_ofs, mode="train")
                    # out is expected to include: 'q_means','p_means','frame_pred','frame_target','loss_recon','loss_sparsity'
                    losses = combined_loss(out, out["frame_target"])
                    loss_all = losses["loss_all"]

                scaler.scale(loss_all).backward()
                # gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

                if step % config["logevery"] == config["logevery"] - 1:
                    # log scalars
                    pbar.set_postfix({'loss': float(loss_all.detach().cpu())})
                    writer.add_scalar('loss_total/train', float(loss_all.detach().cpu()), global_step=step + 1)
                    writer.add_scalar('loss_frame/train', float(losses["loss_frame"].detach().cpu()), global_step=step + 1)
                    writer.add_scalar('loss_kl/train', float(losses["loss_kl"].detach().cpu()), global_step=step + 1)
                    writer.add_scalar('loss_grad/train', float(losses["loss_grad"].detach().cpu()), global_step=step + 1)
                    writer.add_scalar('loss_recon/train', float(losses["loss_recon"].detach().cpu()), global_step=step + 1)
                    writer.add_scalar('loss_sparsity/train', float(losses["loss_sparsity"].detach().cpu()), global_step=step + 1)

                    if "loss_percep" in losses:
                        writer.add_scalar('loss_percep/train', float(losses["loss_percep"].detach().cpu()), global_step=step + 1)
                    if "loss_ssim" in losses:
                        writer.add_scalar('loss_ssim/train', float(losses["loss_ssim"].detach().cpu()), global_step=step + 1)

                    # images
                    num_vis = min(6, sample_frames.size(0))
                    writer.add_figure("img/train_sample_frames",
                                      visualize_sequences(img_batch_tensor2numpy(
                                          sample_frames.cpu()[:num_vis, :, :, :]),
                                          seq_len=sample_frames.size(1) // 3,
                                          return_fig=True),
                                      global_step=step + 1)
                    writer.add_figure("img/train_frame_recon",
                                      visualize_sequences(img_batch_tensor2numpy(
                                          out["frame_pred"].detach().cpu()[:num_vis, :, :, :]),
                                          seq_len=config["model_paras"]["clip_pred"],
                                          return_fig=True),
                                      global_step=step + 1)
                    writer.add_figure("img/train_of_target",
                                      visualize_sequences(img_batch_tensor2numpy(
                                          sample_ofs.cpu()[:num_vis, :, :, :]),
                                          seq_len=sample_ofs.size(1) // 2,
                                          return_fig=True),
                                      global_step=step + 1)
                    writer.add_figure("img/train_of_recon",
                                      visualize_sequences(img_batch_tensor2numpy(
                                          out["of_recon"].detach().cpu()[:num_vis, :, :, :]),
                                          seq_len=sample_ofs.size(1) // 2,
                                          return_fig=True),
                                      global_step=step + 1)

                    writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step=step + 1)

                step += 1
            del dataset
            gc.collect()

        # scheduler per-epoch
        scheduler.step()

        # saving + evaluation
        if epoch % config["saveevery"] == config["saveevery"] - 1:
            model_save_path = os.path.join(paths["ckpt_dir"], config["model_savename"])
            saver(model.state_dict(), optimizer.state_dict(), model_save_path, epoch + 1, step, max_to_save=1)

            stats_save_path = os.path.join(paths["ckpt_dir"], "training_stats.npy-%d" % (epoch + 1))
            cal_training_stats(config, model_save_path + "-%d" % (epoch + 1), training_chunked_samples_dir,
                               stats_save_path)

            with torch.no_grad():
                auc = evaluate(config, model_save_path + "-%d" % (epoch + 1),
                               testing_chunked_samples_file,
                               stats_save_path,
                               suffix=str(epoch + 1))
                print(f'AUC: {auc}')
                if auc > best_auc:
                    best_auc = auc
                    print(f'Best AUC: {best_auc}')
                    only_model_saver(model.state_dict(), os.path.join(paths["ckpt_dir"], "best.pth"))

                writer.add_scalar("auc", auc, global_step=epoch + 1)

    print("================ Best AUC %.4f ================" % best_auc)


def cal_training_stats(config, ckpt_path, training_chunked_samples_dir, stats_save_path):
    device = config["device"]
    model = HFVAD(num_hist=config["model_paras"]["clip_hist"],
                  num_pred=config["model_paras"]["clip_pred"],
                  config=config,
                  features_root=config["model_paras"]["feature_root"],
                  num_slots=config["model_paras"]["num_slots"],
                  shrink_thres=config["model_paras"]["shrink_thres"],
                  mem_usage=config["model_paras"]["mem_usage"],
                  skip_ops=config["model_paras"]["skip_ops"],
                  finetune=config["model_paras"]["finetune"]).to(device).eval()

    model_weights = torch.load(ckpt_path, weights_only=True)["model_state_dict"]
    model.load_state_dict(model_weights)

    score_func = nn.MSELoss(reduction="none")
    training_chunk_samples_files = sorted(os.listdir(training_chunked_samples_dir))

    of_training_stats = []
    frame_training_stats = []

    print("=========Forward pass for training stats ==========")
    with torch.no_grad():
        for chunk_file_idx, chunk_file in enumerate(training_chunk_samples_files):
            dataset = Chunked_sample_dataset(os.path.join(training_chunked_samples_dir, chunk_file))
            dataloader = DataLoader(dataset=dataset, batch_size=128, num_workers=0, shuffle=False)

            for idx, data in tqdm(enumerate(dataloader),
                                  desc="Training stats calculating, Chunked File %02d" % chunk_file_idx,
                                  total=len(dataloader)):
                sample_frames, sample_ofs, _, _, _ = data
                sample_frames = sample_frames.to(device)
                sample_ofs = sample_ofs.to(device)

                out = model(sample_frames, sample_ofs, mode="test")

                loss_frame = score_func(out["frame_pred"], out["frame_target"]).cpu().data.numpy()
                loss_of = score_func(out["of_recon"], out["of_target"]).cpu().data.numpy()

                of_scores = np.sum(np.sum(np.sum(loss_of, axis=3), axis=2), axis=1)
                frame_scores = np.sum(np.sum(np.sum(loss_frame, axis=3), axis=2), axis=1)

                of_training_stats.append(of_scores)
                frame_training_stats.append(frame_scores)
            del dataset
            gc.collect()

    print("=========Forward pass for training stats done!==========")
    of_training_stats = np.concatenate(of_training_stats, axis=0)
    frame_training_stats = np.concatenate(frame_training_stats, axis=0)

    training_stats = dict(of_training_stats=of_training_stats,
                          frame_training_stats=frame_training_stats)
    torch.save(training_stats, stats_save_path)


if __name__ == '__main__':
    config = yaml.safe_load(open("./cfgs/finetune_cfg.yaml"))
    dataset_name = config["dataset_name"]
    dataset_base_dir = config["dataset_base_dir"]
    training_chunked_samples_dir = os.path.join(dataset_base_dir, dataset_name, "training/chunked_samples")
    testing_chunked_samples_file = os.path.join(dataset_base_dir, dataset_name,
                                                "testing/chunked_samples/chunked_samples_00.pkl")

    train(config, training_chunked_samples_dir, testing_chunked_samples_file)