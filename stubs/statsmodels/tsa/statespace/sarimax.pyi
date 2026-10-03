import numpy as np

class SARIMAXResults:
    aic: float
    def forecast(self, steps: int = 1) -> np.ndarray: ...

class SARIMAX:
    def __init__(
        self,
        endog: np.ndarray,
        *,
        order: tuple[int, int, int] = ...,
        seasonal_order: tuple[int, int, int, int] = ...,
        trend: str | None = None,
        enforce_stationarity: bool = ...,
        enforce_invertibility: bool = ...,
    ) -> None: ...
    def fit(self, *, disp: bool = ...) -> SARIMAXResults: ...
