"""
SepFormer training engine.
"""
import os
import csv
import time
import torch
import numpy as np
import scipy.io.wavfile as wav

from loguru import logger
from tqdm import tqdm
from sepformer_utils.decorators import logger_wraps
from sepformer_utils.functions import apply_cmvn
from sepformer_utils.util_engine import (
    load_last_checkpoint_n_get_epoch,
    save_checkpoint_per_best,
    model_params_mac_summary,
)
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from utils.metrics import (
    sisdr, sdr as sdr_metric, sisnr, snr as snr_metric,
    pesq, stoi, compute_bsb_decomposition,
)


def _metrics_to_cpu(tensor_list, lens_list):
    """Convert model output list[K, (B,T)] + lens to numpy for metric computation."""
    K = len(tensor_list)
    B = tensor_list[0].size(0)
    outputs_np = []
    refs_np   = []
    for b in range(B):
        t_len = int(lens_list[b])
        outputs_np.append([tensor_list[k][b, :t_len].cpu().numpy() for k in range(K)])
        refs_np.append([tensor_list[k][b, :t_len].cpu().numpy() for k in range(K)])
    return outputs_np, refs_np


@logger_wraps()
class Engine(object):
    def __init__(self, args, config, model, dataloaders, criterions,
                 optimizers, schedulers, gpuid, device, wandb_run=None):

        self.args = args
        self.engine_mode = args.engine_mode
        self.out_wav_dir  = args.out_wav_dir
        self.config      = config
        self.gpuid       = gpuid
        self.device      = device
        self.wandb_run    = wandb_run
        self.model        = model.to(self.device)
        self.dataloaders  = dataloaders
        self.PIT_SISNR_mag_loss, self.PIT_SISNR_time_loss, \
            self.PIT_SISNRi_loss, self.PIT_SDRi_loss = criterions
        self.main_optimizer = optimizers[0]
        self.main_scheduler, self.warmup_scheduler = schedulers
        self.num_spks = config['model']['num_spks']
        runtime = config.get("runtime", {})

        self.pretrain_weights_path = runtime.get(
            "checkpoints_dir",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "pretrain_weights"),
        )
        os.makedirs(self.pretrain_weights_path, exist_ok=True)
        self.scratch_weights_path = self.pretrain_weights_path
        os.makedirs(self.scratch_weights_path, exist_ok=True)
        self.evaluation_dir = runtime.get("evaluation_dir", os.path.dirname(__file__))
        self.tensorboard_dir = runtime.get(
            "tensorboard_dir",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "tensorboard"),
        )

        self.checkpoint_path = (
            self.pretrain_weights_path
            if any(f.endswith(('.pt', '.pkl', '.pth')) for f in os.listdir(self.pretrain_weights_path))
            else self.scratch_weights_path)
        self.start_epoch = load_last_checkpoint_n_get_epoch(
            self.checkpoint_path, self.model, self.main_optimizer, location=self.device)

        model_params_mac_summary(
            model=self.model,
            input=torch.randn(1, self.config['check_computations']['dummy_len']).to(self.device),
            dummy_input=torch.rand(1, self.config['check_computations']['dummy_len']).to(self.device),
            metrics=['ptflops', 'thop', 'torchinfo']
        )

        logger.info(f"Clip gradient by 2-norm {self.config['engine']['clip_norm']}")

        # Total params for throughput reporting
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.best_val_loss = float("inf")

    @logger_wraps()
    def _train(self, dataloader, epoch):
        self.model.train()
        tot_loss_freq = [0.0] * self.model.num_stages
        tot_loss_time, num_batch = 0.0, 0
        pbar = tqdm(total=len(dataloader),
                    bar_format='{l_bar}{bar:25}{r_bar}{bar:-10b}',
                    colour="YELLOW", dynamic_ncols=True)
        for input_sizes, mixture, src, _ in dataloader:
            nnet_input = mixture
            nnet_input = apply_cmvn(nnet_input) if self.config['engine']['mvn'] else nnet_input
            num_batch += 1
            pbar.update(1)
            if epoch == 1:
                self.warmup_scheduler.step()
            nnet_input = nnet_input.to(self.device)
            self.main_optimizer.zero_grad()
            estim_src, estim_src_bn = torch.nn.parallel.data_parallel(
                self.model, nnet_input, device_ids=self.gpuid)
            cur_loss_s_bn = []
            for idx, estim_src_value in enumerate(estim_src_bn):
                cur_loss_s_bn.append(
                    self.PIT_SISNR_mag_loss(estims=estim_src_value, idx=idx,
                                           input_sizes=input_sizes, target_attr=src))
                tot_loss_freq[idx] += cur_loss_s_bn[idx].item() / self.num_spks
            cur_loss_s = self.PIT_SISNR_time_loss(
                estims=estim_src, input_sizes=input_sizes, target_attr=src)
            tot_loss_time += cur_loss_s.item() / self.num_spks
            alpha = 0.4 * 0.8 ** (1 + (epoch - 101) // 5) if epoch > 100 else 0.4
            cur_loss = ((1 - alpha) * cur_loss_s
                        + alpha * sum(cur_loss_s_bn) / len(cur_loss_s_bn))
            cur_loss = cur_loss / self.num_spks
            cur_loss.backward()
            if self.config['engine']['clip_norm']:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                              self.config['engine']['clip_norm'])
            self.main_optimizer.step()
            dict_loss = {"T_Loss": tot_loss_time / num_batch}
            dict_loss.update({f'F_Loss_{idx}': loss / num_batch
                              for idx, loss in enumerate(tot_loss_freq)})
            pbar.set_postfix(dict_loss)
        pbar.close()
        tot_loss_freq = sum(tot_loss_freq) / len(tot_loss_freq)
        return tot_loss_time / num_batch, tot_loss_freq / num_batch, num_batch

    @logger_wraps()
    def _validate(self, dataloader):
        self.model.eval()
        tot_loss_freq = [0.0] * self.model.num_stages
        tot_loss_time, num_batch = 0.0, 0
        tot_si_sdr = 0.0; tot_sdr = 0.0; tot_si_snr = 0.0; tot_snr = 0.0
        tot_pesq = 0.0; tot_stoi = 0.0
        n_samples = 0

        pbar = tqdm(total=len(dataloader),
                    bar_format='{l_bar}{bar:5}{r_bar}{bar:-10b}',
                    colour="RED", dynamic_ncols=True)
        with torch.inference_mode():
            for input_sizes, mixture, src, _ in dataloader:
                nnet_input = mixture
                nnet_input = apply_cmvn(nnet_input) if self.config['engine']['mvn'] else nnet_input
                nnet_input = nnet_input.to(self.device)
                num_batch += 1
                pbar.update(1)
                estim_src, estim_src_bn = torch.nn.parallel.data_parallel(
                    self.model, nnet_input, device_ids=self.gpuid)

                # ── Loss (keep existing) ─────────────────────────────────────
                cur_loss_s_bn = []
                for idx, estim_src_value in enumerate(estim_src_bn):
                    cur_loss_s_bn.append(
                        self.PIT_SISNR_mag_loss(estims=estim_src_value, idx=idx,
                                               input_sizes=input_sizes, target_attr=src))
                    tot_loss_freq[idx] += cur_loss_s_bn[idx].item() / self.num_spks
                cur_loss_s_SDR = self.PIT_SISNR_time_loss(
                    estims=estim_src, input_sizes=input_sizes, target_attr=src)
                tot_loss_time += cur_loss_s_SDR.item() / self.num_spks

                # ── Metrics (SI-SDR, SDR, SI-SNR, SNR) — cheap ─────────────────
                B = mixture.size(0)
                lens = input_sizes.long().cpu()
                for b in range(B):
                    t_len = int(lens[b])
                    # Speaker 1
                    e1 = estim_src[0][b, :t_len].float()
                    r1 = src[0][b, :t_len].float().to(e1.device)
                    e2 = estim_src[1][b, :t_len].float()
                    r2 = src[1][b, :t_len].float().to(e2.device)

                    s1 = sisdr(e1, r1).item() + sisdr(e2, r2).item()
                    s2 = sdr_metric(e1, r1).item() + sdr_metric(e2, r2).item()
                    s3 = sisnr(e1, r1).item() + sisnr(e2, r2).item()
                    s4 = snr_metric(e1, r1).item() + snr_metric(e2, r2).item()
                    e1_np = e1.detach().cpu().numpy()
                    r1_np = r1.detach().cpu().numpy()
                    e2_np = e2.detach().cpu().numpy()
                    r2_np = r2.detach().cpu().numpy()
                    s5 = pesq(r1_np, e1_np) + pesq(r2_np, e2_np)
                    s6 = stoi(r1_np, e1_np) + stoi(r2_np, e2_np)
                    tot_si_sdr += s1 / 2
                    tot_sdr    += s2 / 2
                    tot_si_snr += s3 / 2
                    tot_snr    += s4 / 2
                    tot_pesq   += s5 / 2
                    tot_stoi   += s6 / 2
                    n_samples  += 1

                dict_loss = {"T_Loss": tot_loss_time / num_batch,
                             "SI-SDR": tot_si_sdr / n_samples,
                             "SDR":    tot_sdr    / n_samples,
                             "PESQ":   tot_pesq   / n_samples,
                             "STOI":   tot_stoi   / n_samples}
                pbar.set_postfix(dict_loss)
        pbar.close()
        tot_loss_freq = sum(tot_loss_freq) / len(tot_loss_freq)

        metrics = {
            "loss_time": tot_loss_time / num_batch,
            "loss_freq": tot_loss_freq / num_batch,
            "si_sdr":    tot_si_sdr / n_samples,
            "sdr":       tot_sdr    / n_samples,
            "si_snr":    tot_si_snr / n_samples,
            "snr":       tot_snr    / n_samples,
            "pesq":      tot_pesq   / n_samples,
            "stoi":      tot_stoi   / n_samples,
        }
        return metrics, num_batch

    @logger_wraps()
    def _test(self, dataloader, wav_dir=None):
        self.model.eval()
        tot_metrics = {k: 0.0 for k in ["si_sdr","sdr","sir","sar","si_snr","snr","pesq","stoi"]}
        num_samples  = 0
        total_rtf    = 0.0   # real-time factor (sec/sec)
        total_rt_sec = 0.0   # absolute seconds
        total_params = self.total_params

        pbar = tqdm(total=len(dataloader),
                    bar_format='{l_bar}{bar:5}{r_bar}{bar:-10b}',
                    colour="grey", dynamic_ncols=True)
        with torch.inference_mode():
            os.makedirs(self.evaluation_dir, exist_ok=True)
            csv_path = os.path.join(self.evaluation_dir, 'test_metrics.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "key","si_sdr","sdr","sir","sar","si_snr","snr","pesq","stoi","rtf","num_params"
                ])

                for input_sizes, mixture, src, key in dataloader:
                    if len(key) > 1:
                        raise RuntimeError("batch size must be 1 for test")

                    nnet_input = mixture.to(self.device)
                    t_sample = int(input_sizes[0].item())
                    t_start  = time.perf_counter()
                    estim_src, _ = torch.nn.parallel.data_parallel(
                        self.model, nnet_input, device_ids=self.gpuid)
                    t_elapsed = time.perf_counter() - t_start

                    # ── Extract per-sample tensors ─────────────────────────────
                    e1 = estim_src[0][0, :t_sample].float().cpu().numpy()
                    e2 = estim_src[1][0, :t_sample].float().cpu().numpy()
                    r1 = src[0][0, :t_sample].float().cpu().numpy()
                    r2 = src[1][0, :t_sample].float().cpu().numpy()

                    estim_list = [e1, e2]
                    ref_list   = [r1, r2]

                    # ── Cheap metrics ──────────────────────────────────────────
                    e1_t = torch.from_numpy(e1); r1_t = torch.from_numpy(r1)
                    e2_t = torch.from_numpy(e2); r2_t = torch.from_numpy(r2)
                    m_si_sdr = (sisdr(e1_t, r1_t).item() + sisdr(e2_t, r2_t).item()) / 2
                    m_sdr    = (sdr_metric(e1_t, r1_t).item() + sdr_metric(e2_t, r2_t).item()) / 2
                    m_si_snr = (sisnr(e1_t, r1_t).item() + sisnr(e2_t, r2_t).item()) / 2
                    m_snr    = (snr_metric(e1_t, r1_t).item() + snr_metric(e2_t, r2_t).item()) / 2

                    # ── BSS-EVAL metrics (SIR, SAR) ──────────────────────────────
                    bss_sdr, bss_sir, bss_sar = compute_bsb_decomposition(ref_list, estim_list)
                    m_sdr2 = float(np.mean(bss_sdr))
                    m_sir  = float(np.mean(bss_sir))
                    m_sar  = float(np.mean(bss_sar))

                    # ── PESQ + STOI ─────────────────────────────────────────────
                    m_pesq = (pesq(r1, e1) + pesq(r2, e2)) / 2
                    m_stoi = (stoi(r1, e1) + stoi(r2, e2)) / 2

                    # ── RTF ─────────────────────────────────────────────────────
                    rtf = t_elapsed / (t_sample / 8000)  # sec inference / sec audio
                    total_rt_sec += t_elapsed

                    # Accumulate
                    tot_metrics["si_sdr"] += m_si_sdr
                    tot_metrics["sdr"]    += (m_sdr + m_sdr2) / 2
                    tot_metrics["sir"]    += m_sir
                    tot_metrics["sar"]    += m_sar
                    tot_metrics["si_snr"] += m_si_snr
                    tot_metrics["snr"]    += m_snr
                    tot_metrics["pesq"]   += m_pesq
                    tot_metrics["stoi"]   += m_stoi
                    num_samples += 1
                    total_rtf   += rtf

                    # Per-sample CSV
                    writer.writerow([
                        key[0][:-4],
                        f"{m_si_sdr:.4f}", f"{m_sdr:.4f}",
                        f"{m_sir:.4f}", f"{m_sar:.4f}",
                        f"{m_si_snr:.4f}", f"{m_snr:.4f}",
                        f"{m_pesq:.4f}", f"{m_stoi:.4f}",
                        f"{rtf:.6f}", f"{total_params:,}"
                    ])

                    # Save wav
                    if self.engine_mode == "test_save":
                        if wav_dir is None:
                            wav_dir = self.args.out_wav_dir or os.path.join(self.evaluation_dir, "wav_out")
                        os.makedirs(wav_dir, exist_ok=True)
                        mix_np = mixture[0, :t_sample].cpu().numpy()
                        for name, arr in [("mixture", mix_np),
                                          (f"{key[0][:-4]}_spk1", e1),
                                          (f"{key[0][:-4]}_spk2", e2)]:
                            norm = np.clip(0.5 * arr / max(abs(arr).max(), 1e-8), -1, 1)
                            wav.write(os.path.join(wav_dir, f"{name}.wav"),
                                      8000, (norm * 32767).astype(np.int16))

                    pbar.set_postfix({
                        "SI-SDR": tot_metrics["si_sdr"] / num_samples,
                        "SDR":    tot_metrics["sdr"]    / num_samples,
                        "PESQ":   tot_metrics["pesq"]   / num_samples,
                        "STOI":   tot_metrics["stoi"]   / num_samples,
                        "RTF":    total_rtf / num_samples,
                    })
                    pbar.update(1)

        pbar.close()

        # ── Summary ──────────────────────────────────────────────────────────
        avg = {k: v / num_samples for k, v in tot_metrics.items()}
        avg_rtf = total_rtf / num_samples
        summary = {
            "SI-SDR (dB)": avg["si_sdr"],
            "SDR     (dB)": avg["sdr"],
            "SIR     (dB)": avg["sir"],
            "SAR     (dB)": avg["sar"],
            "SI-SNR  (dB)": avg["si_snr"],
            "SNR     (dB)": avg["snr"],
            "PESQ        ": avg["pesq"],
            "STOI        ": avg["stoi"],
            "RTF         ": avg_rtf,
            "#params(M)  ": total_params / 1e6,
            "rt_sec      ": total_rt_sec,
        }
        for k, v in summary.items():
            logger.info(f"  {k:15s}: {v}")
        return summary, csv_path

    @logger_wraps()
    def run(self):
        writer_src = SummaryWriter(self.tensorboard_dir)

        if "test" in self.engine_mode:
            t0 = time.time()
            _, csv_path = self._test(self.dataloaders['test'], self.out_wav_dir)
            t1 = time.time()
            logger.info(f"[TEST] done in {t1 - t0:.1f}s | CSV → {csv_path}")
            logger.info("Testing done!")
            writer_src.close()
            return

        t_init = time.time()
        if self.start_epoch > 1:
            val_metrics, _ = self._validate(self.dataloaders['valid'])
            init_loss_time = val_metrics["loss_time"]
            init_loss_freq = val_metrics["loss_freq"]
        else:
            init_loss_time = init_loss_freq = 0.0
        logger.info(f"[INIT] Loss_t={init_loss_time:.4f} | Loss_f={init_loss_freq:.4f} "
                   f"| took {time.time()-t_init:.1f}s")

        for epoch in range(self.start_epoch, self.config['engine']['max_epoch'] + 1):
            t_train = time.time()
            train_loss_time, train_loss_freq, _ = self._train(self.dataloaders['train'], epoch)
            t_train = time.time() - t_train

            t_valid = time.time()
            val_metrics, _ = self._validate(self.dataloaders['valid'])
            t_valid = time.time() - t_valid
            val_loss_time = val_metrics["loss_time"]
            val_loss_freq = val_metrics["loss_freq"]

            if epoch > self.config['engine']['start_scheduling']:
                self.main_scheduler.step(val_loss_time)

            logger.info(
                f"[{epoch:3d}] T_t={train_loss_time:.4f} T_f={train_loss_freq:.4f} "
                f"V_t={val_loss_time:.4f} V_f={val_loss_freq:.4f} "
                f"SI-SDR={val_metrics['si_sdr']:.2f}dB SDR={val_metrics['sdr']:.2f}dB "
                f"PESQ={val_metrics['pesq']:.3f} STOI={val_metrics['stoi']:.3f} | "
                f"t_train={t_train:.1f}s t_valid={t_valid:.1f}s")

            self.best_val_loss = save_checkpoint_per_best(
                self.best_val_loss,
                val_loss_time,
                train_loss_time,
                epoch,
                self.model,
                self.main_optimizer,
                self.checkpoint_path,
            )

            writer_src.add_scalars("Loss", {
                'train_time': train_loss_time,
                'valid_time': val_loss_time,
                'train_freq': train_loss_freq,
                'valid_freq': val_loss_freq}, epoch)
            writer_src.add_scalar("Metrics/valid_si_sdr", val_metrics['si_sdr'], epoch)
            writer_src.add_scalar("Metrics/valid_sdr", val_metrics['sdr'], epoch)
            writer_src.add_scalar("Metrics/valid_pesq", val_metrics['pesq'], epoch)
            writer_src.add_scalar("Metrics/valid_stoi", val_metrics['stoi'], epoch)
            writer_src.add_scalar("LR", self.main_optimizer.param_groups[0]['lr'], epoch)
            writer_src.flush()

            if self.wandb_run is not None:
                self.wandb_run.log({
                    "epoch":             epoch,
                    "train/loss_time":   train_loss_time,
                    "train/loss_freq":   train_loss_freq,
                    "valid/loss_time":   val_loss_time,
                    "valid/loss_freq":   val_loss_freq,
                    "valid/si_sdr":      val_metrics['si_sdr'],
                    "valid/sdr":         val_metrics['sdr'],
                    "valid/si_snr":      val_metrics['si_snr'],
                    "valid/snr":         val_metrics['snr'],
                    "valid/pesq":        val_metrics['pesq'],
                    "valid/stoi":        val_metrics['stoi'],
                    "train/sisnr_db":    -train_loss_time,
                    "valid/sisnr_db":    -val_loss_time,
                    "learning_rate":     self.main_optimizer.param_groups[0]['lr'],
                })

            if epoch in self.config['engine']['test_epochs']:
                t_test = time.time()
                test_summary, _ = self._test(self.dataloaders['test'])
                t_test = time.time() - t_test
                logger.info(f"[TEST @ epoch {epoch}] done in {t_test:.1f}s")
                if self.wandb_run is not None:
                    self.wandb_run.log({f"test/{k}": v for k, v in test_summary.items()}, step=epoch)

        writer_src.close()
        logger.info(f"Training for {self.config['engine']['max_epoch']} epochs done!")
