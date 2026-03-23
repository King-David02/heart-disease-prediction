import pandas as pd
from abc import ABC, abstractmethod

class BaseModel(ABC):
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass
    
    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        pass