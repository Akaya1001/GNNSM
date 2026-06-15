"""Shared utilities: device selection and evaluation metrics."""

from .device import get_device, device_report
from .metrics import evaluate_f1

__all__ = ["get_device", "device_report", "evaluate_f1"]
