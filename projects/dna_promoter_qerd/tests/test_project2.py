"""Unit and end-to-end integration tests for Project 2 (`dna_promoter_qerd`)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from projects.dna_promoter_qerd.biology import compute_santalucia_dg37
from projects.dna_promoter_qerd.cli import main as cli_main
from projects.dna_promoter_qerd.loaders import seqs_to_spins, spins_to_seqs
from projects.dna_promoter_qerd.hardware import get_fake_heron, select_clean_dna_subgraph


class TestProject2Suite(unittest.TestCase):
    def test_dna_bijection_thermo_and_clean_128q_subgraph(self):
        seqs = ["GGGCCCGGGTATAAAAGGGCACACACACACAGTCAGTCAGTCAATGCTAGCTAGCTAGCATCGA"]
        spins = seqs_to_spins(seqs)
        self.assertEqual(spins.shape, (1, 128))
        self.assertEqual(spins_to_seqs(spins), seqs)
        self.assertLess(compute_santalucia_dg37(seqs[0]), -70.0)

        fk = get_fake_heron("ibm_kingston")
        edges, phys, rep = select_clean_dna_subgraph(fk, n_qubits=128)
        self.assertEqual(len(phys), 128)
        self.assertGreaterEqual(rep.intra_base_coupled, 40)
        self.assertLessEqual(rep.bad_cz_gt2pct, 4)

    def test_quick_preset_end_to_end_and_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            art_dir = os.path.join(tmp, "artifacts")
            ckpt_dir = os.path.join(tmp, "checkpoints")
            cli_main(["--preset", "quick", "--skip-scaling", "--artifact-dir", art_dir, "--ckpt-dir", ckpt_dir])
            m_path = os.path.join(art_dir, "metrics.json")
            self.assertTrue(os.path.exists(m_path))
            with open(m_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            self.assertGreater(m["oracle_cnn_metrics"]["test_auc"], 0.85)
            self.assertIn("qerd_qpu", m["comparison"])
            self.assertGreater(
                m["comparison"]["qerd_qpu"]["valid_promoter_pct"],
                m["comparison"]["untrained_qpu"]["valid_promoter_pct"],
            )
            self.assertTrue(os.path.exists(os.path.join(art_dir, "fig_dna_qpu_showcase.png")))
            self.assertTrue(os.path.exists(os.path.join(art_dir, "generated_promoters.fasta")))
            cli_main(["--preset", "quick", "--infer", "--ckpt-dir", ckpt_dir])


if __name__ == "__main__":
    unittest.main()
