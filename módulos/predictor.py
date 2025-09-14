import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
from .data_collector import DataCollector
from .indicators import TechnicalIndicators
from .ml_models import MLModels
from .storage import DataStorage
from config import CURRENCY_PAIRS, PREDICTION_INTERVALS, CONFIDENCE_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForexPredictor:
    def __init__(self):
        self.data_collector = DataCollector()
        self.storage = DataStorage()
        self.models = {}  # Un modelo por par de divisas
        
    def initialize_models(self, pairs: List[str] = None):
        """
        Inicializa modelos para los pares especificados
        """
        if pairs is None:
            pairs = CURRENCY_PAIRS
        
        for pair in pairs:
            logger.info(f"Inicializando modelo para {pair}")
            self.models[pair] = MLModels()
            
            # Intentar cargar modelo existente
            if not self.models[pair].load_models(pair):
                # Si no existe, entrenar nuevo modelo
                logger.info(f"Entrenando nuevo modelo para {pair}")
                self.train_model(pair)
    
    def train_model(self, symbol: str):
        """
        Entrena modelo para un par específico
        """
        try:
            #
