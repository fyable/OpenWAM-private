import os
import unittest
from unittest.mock import MagicMock, patch

from benchmarks.video_parallel.gpu_wait import available, main, reserve, snapshot


class GpuWaitTests(unittest.TestCase):
    def test_snapshot_and_busy_filter(self):
        with patch("subprocess.check_output", side_effect=[
            "0, GPU-first, 100, 0\n1, GPU-second, 200, 1\n", "GPU-second\n"
        ]):
            devices, busy = snapshot()
        self.assertEqual(available(devices, busy, 1024, 5), {"0"})

    def test_thresholds_and_unknown_metrics(self):
        devices = {"0": ("a", 2048, 0), "1": ("b", 0, 10), "2": ("c", float("nan"), 0)}
        self.assertEqual(available(devices, set(), 1024, 5), set())

    def test_reservation(self):
        locks = reserve([f"test-{os.getpid()}"])
        try:
            self.assertEqual(len(locks), 1)
            self.assertEqual(reserve([f"test-{os.getpid()}"]), [])
        finally:
            for handle in locks:
                handle.close()

    def test_auto_wait_then_exec(self):
        devices = {"2": ("GPU-test", 10, 0)}
        with patch.dict(os.environ, {}, clear=True), \
                patch("sys.argv", ["wait", "--idle-seconds", "0", "--", "bash", "job.sh"]), \
                patch("benchmarks.video_parallel.gpu_wait.snapshot", side_effect=[
                    (devices, {"GPU-test"}), (devices, set()), (devices, set())]), \
                patch("benchmarks.video_parallel.gpu_wait.reserve", return_value=[MagicMock()]), \
                patch("time.sleep") as sleep, patch("os.set_inheritable"), \
                patch("os.execvpe", side_effect=RuntimeError("exec sentinel")) as execute:
            with self.assertRaisesRegex(RuntimeError, "exec sentinel"):
                main()
        sleep.assert_called_once()
        environment = execute.call_args.args[2]
        self.assertEqual(environment["GPU"], "2")
        self.assertEqual(environment["GPUS"], "2")
        self.assertEqual(environment["DEVICE"], "cuda:2")
        self.assertEqual(environment["WAIT_FOR_GPU"], "0")

    def test_timeout(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch("sys.argv", ["wait", "--timeout", "1", "--", "bash", "job.sh"]), \
                patch("time.monotonic", side_effect=[0, 2]), patch("os.execvpe") as execute:
            with self.assertRaises(TimeoutError):
                main()
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
