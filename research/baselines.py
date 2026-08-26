from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


@dataclass
class PanTompkinsStyleDetector:
    """Transparent Pan-Tompkins-style QRS baseline.

    This compact implementation follows the main processing stages of the
    classical method but is intentionally labeled "style" rather than claiming
    byte-for-byte equivalence with the original 1985 implementation.
    """

    low_hz: float = 5.0
    high_hz: float = 15.0
    integration_ms: float = 150.0
    refractory_ms: float = 200.0
    refinement_ms: float = 80.0

    def detect(self, signal: np.ndarray, fs: float) -> np.ndarray:
        x = np.asarray(signal, dtype=float)
        if x.ndim != 1:
            raise ValueError("signal must be one-dimensional")
        if x.size < max(32, int(fs)):
            return np.array([], dtype=int)

        nyq = fs / 2.0
        high = min(self.high_hz, 0.90 * nyq)
        low = min(self.low_hz, high * 0.6)
        sos = butter(2, [low / nyq, high / nyq], btype="bandpass", output="sos")
        y = sosfiltfilt(sos, x)
        derivative = np.gradient(y)
        squared = derivative * derivative
        win = max(1, int(round(self.integration_ms * fs / 1000.0)))
        integrated = np.convolve(squared, np.ones(win) / win, mode="same")

        candidates, _ = find_peaks(
            integrated,
            distance=max(1, int(round(0.12 * fs))),
        )
        if candidates.size == 0:
            return np.array([], dtype=int)

        init_n = min(len(integrated), max(1, int(round(2.0 * fs))))
        initial = integrated[:init_n]
        npki = float(np.percentile(initial, 50))
        spki = float(np.percentile(initial, 90))
        refractory = max(1, int(round(self.refractory_ms * fs / 1000.0)))
        accepted: list[int] = []

        for candidate in candidates:
            amp = float(integrated[candidate])
            threshold = npki + 0.25 * (spki - npki)
            if amp >= threshold and (
                not accepted or int(candidate) - accepted[-1] >= refractory
            ):
                accepted.append(int(candidate))
                spki = 0.125 * amp + 0.875 * spki
            else:
                npki = 0.125 * amp + 0.875 * npki

        refine = max(1, int(round(self.refinement_ms * fs / 1000.0)))
        peaks: list[int] = []
        for candidate in accepted:
            lo = max(0, candidate - refine)
            hi = min(x.size, candidate + refine + 1)
            local = x[lo:hi]
            baseline = np.median(local)
            r_peak = lo + int(np.argmax(np.abs(local - baseline)))
            if not peaks or r_peak - peaks[-1] >= refractory:
                peaks.append(r_peak)
        return np.asarray(peaks, dtype=int)
