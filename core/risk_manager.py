"""
Risk Management Module - Sistema profesional de gestión de riesgo
Incluye cálculo de stops/targets dinámicos basados en ATR
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class RiskManager:
    """Gestión avanzada de riesgo con stops/targets dinámicos"""
    
    def __init__(self, 
                 default_risk_pct: float = 2.0,  # % del capital en riesgo por trade
                 risk_reward_ratio: float = 2.0,  # Ratio mínimo risk/reward
                 max_position_size_pct: float = 10.0):  # Máximo % del portfolio por activo
        self.default_risk_pct = default_risk_pct
        self.risk_reward_ratio = risk_reward_ratio
        self.max_position_size_pct = max_position_size_pct
    
    def calculate_atr_stops(self, 
                           data: pd.DataFrame,
                           entry_price: float,
                           atr_multiplier_stop: float = 2.0,
                           atr_multiplier_target: float = 3.0) -> Dict:
        """
        Calcula stops y targets dinámicos basados en ATR
        
        Returns:
            Dict con stop_loss, take_profit_1, take_profit_2, riesgo_reward
        """
        if 'ATR' not in data.columns:
            # Calcular ATR si no existe
            high_low = data['High'] - data['Low']
            high_close = (data['High'] - data['Close'].shift()).abs()
            low_close = (data['Low'] - data['Close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.ewm(alpha=1/14, adjust=False).mean()
        else:
            atr = data['ATR']
        
        current_atr = atr.iloc[-1]
        
        # Calcular niveles
        stop_loss = entry_price - (current_atr * atr_multiplier_stop)
        take_profit_1 = entry_price + (current_atr * atr_multiplier_target)
        take_profit_2 = entry_price + (current_atr * atr_multiplier_target * 2)
        
        # Calcular risk/reward
        risk = entry_price - stop_loss
        reward = take_profit_1 - entry_price
        rr_ratio = reward / risk if risk > 0 else 0
        
        return {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'stop_loss_pct': ((stop_loss - entry_price) / entry_price) * 100,
            'take_profit_1': take_profit_1,
            'take_profit_1_pct': ((take_profit_1 - entry_price) / entry_price) * 100,
            'take_profit_2': take_profit_2,
            'take_profit_2_pct': ((take_profit_2 - entry_price) / entry_price) * 100,
            'atr_value': current_atr,
            'risk_reward_ratio': rr_ratio,
            'risk_amount': risk,
            'reward_amount': reward
        }
    
    def calculate_position_size(self,
                               account_size: float,
                               entry_price: float,
                               stop_loss: float,
                               risk_pct: Optional[float] = None) -> Dict:
        """
        Calcula el tamaño de posición óptimo basado en riesgo por trade
        
        Args:
            account_size: Capital total disponible
            entry_price: Precio de entrada
            stop_loss: Nivel de stop loss
            risk_pct: % del capital a arriesgar (default: self.default_risk_pct)
        
        Returns:
            Dict con shares, position_value, risk_amount, max_loss_pct
        """
        if risk_pct is None:
            risk_pct = self.default_risk_pct
        
        # Cantidad a arriesgar
        risk_amount = account_size * (risk_pct / 100)
        
        # Riesgo por acción
        risk_per_share = abs(entry_price - stop_loss)
        
        # Calcular número de acciones
        shares = risk_amount / risk_per_share if risk_per_share > 0 else 0
        
        # Valor de la posición
        position_value = shares * entry_price
        
        # Verificar que no exceda el máximo permitido
        max_position_value = account_size * (self.max_position_size_pct / 100)
        
        if position_value > max_position_value:
            shares = max_position_value / entry_price
            position_value = max_position_value
        
        # Pérdida máxima
        max_loss = shares * risk_per_share
        max_loss_pct = (max_loss / account_size) * 100
        
        return {
            'shares': int(shares),
            'position_value': position_value,
            'position_size_pct': (position_value / account_size) * 100,
            'risk_amount': risk_amount,
            'risk_per_share': risk_per_share,
            'max_loss': max_loss,
            'max_loss_pct': max_loss_pct,
            'is_within_limits': position_value <= max_position_value
        }
    
    def calculate_kelly_criterion(self,
                                  win_rate: float,
                                  avg_win: float,
                                  avg_loss: float) -> float:
        """
        Calcula el Kelly Criterion para sizing óptimo
        
        Args:
            win_rate: Tasa de ganancia (0-1)
            avg_win: Ganancia promedio por trade ganador
            avg_loss: Pérdida promedio por trade perdedor
        
        Returns:
            Kelly percentage (0-1)
        """
        if avg_loss == 0 or win_rate >= 1 or win_rate <= 0:
            return 0.0
        
        b = avg_win / abs(avg_loss)  # win/loss ratio
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Aplicar fracción conservadora (1/2 Kelly)
        return max(0, min(kelly * 0.5, 0.25))  # Máximo 25%
    
    def trailing_stop(self,
                     current_price: float,
                     entry_price: float,
                     highest_price: float,
                     atr: float,
                     trailing_multiplier: float = 2.0) -> Dict:
        """
        Calcula trailing stop dinámico basado en ATR
        
        Returns:
            Dict con nuevo stop_loss y si debe activarse
        """
        # Calcular ganancia actual
        current_profit_pct = ((current_price - entry_price) / entry_price) * 100
        
        # Solo activar trailing stop si hay ganancia mínima
        if current_profit_pct < 5:
            return {
                'trailing_active': False,
                'stop_loss': entry_price - (atr * trailing_multiplier),
                'reason': 'Ganancia insuficiente para trailing'
            }
        
        # Trailing stop desde el máximo alcanzado
        new_stop = highest_price - (atr * trailing_multiplier)
        
        # El stop solo puede subir, nunca bajar
        original_stop = entry_price - (atr * trailing_multiplier)
        final_stop = max(new_stop, original_stop)
        
        return {
            'trailing_active': True,
            'stop_loss': final_stop,
            'stop_pct_from_entry': ((final_stop - entry_price) / entry_price) * 100,
            'protection_pct': ((final_stop - entry_price) / (highest_price - entry_price)) * 100,
            'reason': 'Trailing activado - protegiendo ganancias'
        }
    
    def volatility_adjusted_position(self,
                                    base_position_size: float,
                                    current_volatility: float,
                                    avg_volatility: float) -> float:
        """
        Ajusta el tamaño de posición basado en volatilidad actual vs promedio
        
        Args:
            base_position_size: Tamaño base de posición
            current_volatility: ATR actual
            avg_volatility: ATR promedio histórico
        
        Returns:
            Tamaño ajustado de posición
        """
        if avg_volatility == 0:
            return base_position_size
        
        # Si volatilidad es mayor que promedio, reducir posición
        volatility_ratio = current_volatility / avg_volatility
        
        if volatility_ratio > 1.5:  # Volatilidad muy alta
            adjustment = 0.5  # Reducir 50%
        elif volatility_ratio > 1.2:  # Volatilidad moderadamente alta
            adjustment = 0.75  # Reducir 25%
        elif volatility_ratio < 0.8:  # Volatilidad baja
            adjustment = 1.2  # Aumentar 20%
        else:
            adjustment = 1.0  # Sin cambios
        
        return base_position_size * adjustment
    
    def portfolio_heat(self, 
                      open_positions: list,
                      account_size: float) -> Dict:
        """
        Calcula el "calor" del portfolio (riesgo total actual)
        
        Args:
            open_positions: Lista de posiciones abiertas con su riesgo
            account_size: Capital total
        
        Returns:
            Dict con métricas de riesgo agregado
        """
        total_risk = sum(pos.get('risk_amount', 0) for pos in open_positions)
        total_exposure = sum(pos.get('position_value', 0) for pos in open_positions)
        
        risk_pct = (total_risk / account_size) * 100
        exposure_pct = (total_exposure / account_size) * 100
        
        # Semáforo de riesgo
        if risk_pct > 10:
            risk_level = 'CRITICAL'
            color = '🔴'
        elif risk_pct > 6:
            risk_level = 'HIGH'
            color = '🟠'
        elif risk_pct > 3:
            risk_level = 'MEDIUM'
            color = '🟡'
        else:
            risk_level = 'LOW'
            color = '🟢'
        
        return {
            'total_risk': total_risk,
            'total_risk_pct': risk_pct,
            'total_exposure': total_exposure,
            'total_exposure_pct': exposure_pct,
            'num_positions': len(open_positions),
            'risk_level': risk_level,
            'risk_indicator': color,
            'can_add_position': risk_pct < 8  # No agregar si riesgo > 8%
        }
