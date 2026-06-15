"""AVICI trains for a few steps on CPU; loss is finite and decreases."""
import torch

from meterhierarchy.avici.train import run_smoke


def test_avici_smoke_cpu():
    run_smoke(torch.device("cpu"), log=lambda *_a: None)


if __name__ == "__main__":
    test_avici_smoke_cpu()
    print("test_smoke_avici OK")
