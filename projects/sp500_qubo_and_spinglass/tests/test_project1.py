"""Unit and integration tests for `projects.sp500_qubo_and_spinglass`."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from projects.sp500_qubo_and_spinglass.configs import build_run_config
from projects.sp500_qubo_and_spinglass.hardware import audit_all_heron_r2, project_heron_timing, select_clean_subgraph
from projects.sp500_qubo_and_spinglass.hardware.heron_r2 import get_fake_heron
from projects.sp500_qubo_and_spinglass.reporting import format_heron_comparison
from projects.sp500_qubo_and_spinglass.cli import main as cli_main


class TestProject1Suite(unittest.TestCase):
    def test_heron_r2_calibration_ranking_and_4k_projection(self):
        audits = audit_all_heron_r2()
        names = [a.device for a in audits]
        self.assertEqual(set(names), {"ibm_kingston", "ibm_fez", "ibm_marrakesh"})
        self.assertEqual(names[0], "ibm_kingston")
        proj = project_heron_timing(156, 176, n_steps=200, batch_size=10, shots_per_run=20, shots_per_sec=4000.0)
        self.assertEqual(proj.qerd_shots, 80_000)
        self.assertAlmostEqual(proj.qerd_qpu_pulse_sec, 20.0, places=2)
        self.assertGreater(proj.psr_equal_steps_pulse_sec, 6000.0)
        txt = format_heron_comparison(audits, proj)
        self.assertIn("ibm_kingston", txt)

    def test_heron_128q_sweet_spot_and_38k_budget(self):
        cfg = build_run_config("heron_128q")
        self.assertEqual(cfg.n_qubits, 128)
        self.assertEqual(cfg.max_shots, 38_400)
        self.assertEqual(cfg.cal_mode, "inverse")
        self.assertEqual(cfg.nudge_mode, "residual")
        self.assertTrue(cfg.readout_mitigation)

        fk = get_fake_heron("ibm_kingston")
        _, phys128, rep128 = select_clean_subgraph(fk, n_qubits=128)
        self.assertLess(rep128.bad_ro_gt5pct, 16)
        self.assertLessEqual(rep128.bad_cz_gt2pct, 2)
        _, _, rep32 = select_clean_subgraph(fk, n_qubits=32, allowed_phys=phys128)
        self.assertEqual(rep32.bad_ro_gt5pct, 0)
        self.assertEqual(rep32.bad_cz_gt2pct, 0)

        cfg_sg = build_run_config("spinglass_only_128q")
        self.assertEqual(cfg_sg.n_qubits, 128)
        self.assertTrue(cfg_sg.run_live_spsa)
        self.assertTrue(cfg_sg.run_live_nozz)
        self.assertTrue(cfg_sg.skip_minvar)
        self.assertFalse(cfg_sg.skip_untrained)

    def test_quick_preset_end_to_end_and_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            art_dir = os.path.join(tmp, "artifacts")
            ckpt_dir = os.path.join(tmp, "checkpoints")
            cli_main(["--preset", "quick", "--skip-scaling", "--artifact-dir", art_dir, "--ckpt-dir", ckpt_dir])
            m_path = os.path.join(art_dir, "metrics.json")
            self.assertTrue(os.path.exists(m_path))
            with open(m_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            self.assertIn("buy_hold_1N", m["sp500_minvar_qubo"])
            self.assertIn("oos_ens_profit_usd", m["sp500_minvar_qubo"]["buy_hold_1N"])
            self.assertGreater(m["sp500_minvar_qubo"]["qerd_qpu"]["mean_gain"], 0.45)
            self.assertGreater(m["heavy_hex_spinglass"]["qerd_qpu"]["mean_gain"], 0.45)
            self.assertIn("subgraph_audits", m)
            self.assertTrue(os.path.exists(os.path.join(art_dir, "fig_hardware_advantage.png")))
            self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "minvar_qerd_8q.npz")))
            self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "spinglass_qerd_8q.npz")))
            cli_main(["--preset", "quick", "--infer", "minvar", "--ckpt-dir", ckpt_dir])
            cli_main(["--preset", "quick", "--infer", "spinglass", "--ckpt-dir", ckpt_dir])


if __name__ == "__main__":
    unittest.main()
