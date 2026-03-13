"""
CONSENSUS SCORING SYSTEM — v2.0 (Pesos Dinámicos)
Combina todos los análisis (Técnico, ML, LSTM, Groq) en una recomendación
única ponderada. Los pesos se ajustan automáticamente según:

  1. Régimen de mercado (VIX alto → más peso a Técnico, menos a ML)
  2. Calidad del ML    (accuracy real post-leakage-fix → escala su peso)
  3. Acuerdo entre fuentes (alta divergencia → reduce confianza y ajusta pesos)
  4. Disponibilidad   (si LSTM no está entrenado, su peso se redistribuye)
"""

import numpy as np
from typing import Dict, Optional, List


# ============================================================================
# TABLAS DE PESOS DINÁMICOS POR RÉGIMEN
# ============================================================================
# Lógica:
#   RISK_ON  (VIX < 19): mercado tranquilo, ML y LSTM son más fiables
#   NEUTRAL  (VIX 19-25): pesos balanceados
#   RISK_OFF (VIX 25-35): volatilidad alta, Técnico y Groq pesan más
#   CRISIS   (VIX > 35):  solo confiar en señales técnicas y Groq

REGIME_WEIGHTS = {
    'RISK_ON': {
        'technical': 25,
        'ml':        30,   # ML más fiable en mercados tranquilos
        'lstm':      30,   # LSTM idem
        'groq':      15,
    },
    'NEUTRAL': {
        'technical': 30,
        'ml':        25,
        'lstm':      25,
        'groq':      20,
    },
    'RISK_OFF': {
        'technical': 40,   # Indicadores técnicos más confiables en alta vol
        'ml':        15,   # ML entrenado en datos "normales" pierde precisión
        'lstm':      15,
        'groq':      30,   # Groq puede contextualizar eventos macro
    },
    'CRISIS': {
        'technical': 50,
        'ml':        10,
        'lstm':      10,
        'groq':      30,
    },
}

# Umbrales de VIX para detectar régimen
VIX_THRESHOLDS = {
    'RISK_ON':  (0,  19),
    'NEUTRAL':  (19, 25),
    'RISK_OFF': (25, 35),
    'CRISIS':   (35, 999),
}


class ConsensusAnalyzer:
    """
    Combina múltiples fuentes de análisis en un score de consenso ponderado.
    Los pesos se ajustan dinámicamente según el contexto de mercado.
    """

    def __init__(self, weights: Optional[Dict] = None):
        """
        Args:
            weights: Pesos manuales opcionales. Si None, se calculan dinámicamente.
        """
        self._manual_weights = weights  # guardamos para referencia
        # Pesos base (se sobreescriben en analyze_consensus si hay contexto)
        self.weights = weights or dict(REGIME_WEIGHTS['NEUTRAL'])

    # =========================================================================
    # MÉTODO PRINCIPAL
    # =========================================================================

    def analyze_consensus(self,
                          technical_score: float,
                          ml_prediction:   Optional[Dict] = None,
                          lstm_prediction:  Optional[Dict] = None,
                          groq_analysis:    Optional[str]  = None,
                          market_context:   Optional[Dict] = None,
                          ml_accuracy:      Optional[float] = None) -> Dict:
        """
        Genera consensus score combinando todas las fuentes disponibles.

        Args:
            technical_score : Score técnico (-100 a +100)
            ml_prediction   : Dict con 'probability_up' y opcionalmente 'confidence'
            lstm_prediction : Dict con 'probability_up'
            groq_analysis   : Texto del análisis de Groq
            market_context  : Dict con 'vix' (float) y 'regime' (str) del fetcher
            ml_accuracy     : Accuracy real del modelo ML (0-1), post corrección leakage

        Returns:
            Dict con consensus_score, confidence, recommendation, breakdown, etc.
        """
        # ------------------------------------------------------------------
        # 1. CALCULAR PESOS DINÁMICOS
        # ------------------------------------------------------------------
        dynamic_weights = self._compute_dynamic_weights(
            ml_prediction   = ml_prediction,
            lstm_prediction  = lstm_prediction,
            groq_analysis   = groq_analysis,
            market_context  = market_context,
            ml_accuracy     = ml_accuracy,
        )

        # ------------------------------------------------------------------
        # 2. NORMALIZAR CADA FUENTE A ESCALA 0-100
        # ------------------------------------------------------------------
        scores: Dict[str, float] = {}
        available: List[str] = []

        # Técnico: -100..+100 → 0..100
        tech_norm = ((technical_score + 100) / 200) * 100
        scores['technical'] = float(np.clip(tech_norm, 0, 100))
        available.append('technical')

        if ml_prediction and ml_prediction.get('probability_up') is not None:
            scores['ml'] = float(np.clip(ml_prediction['probability_up'] * 100, 0, 100))
            available.append('ml')

        if lstm_prediction and lstm_prediction.get('probability_up') is not None:
            scores['lstm'] = float(np.clip(lstm_prediction['probability_up'] * 100, 0, 100))
            available.append('lstm')

        if groq_analysis:
            scores['groq'] = float(np.clip(self._extract_groq_sentiment(groq_analysis), 0, 100))
            available.append('groq')

        # ------------------------------------------------------------------
        # 3. CONSENSUS SCORE PONDERADO
        # ------------------------------------------------------------------
        total_w = sum(dynamic_weights[s] for s in available)
        consensus_score = sum(
            scores[s] * (dynamic_weights[s] / total_w)
            for s in available
        )

        # ------------------------------------------------------------------
        # 4. CONFIANZA
        # ------------------------------------------------------------------
        confidence = self._calculate_confidence(scores, available, dynamic_weights)

        # ------------------------------------------------------------------
        # 5. RECOMENDACIÓN
        # ------------------------------------------------------------------
        recommendation = self._get_recommendation(consensus_score, confidence)

        # ------------------------------------------------------------------
        # 6. DISCREPANCIAS
        # ------------------------------------------------------------------
        discrepancies = self._analyze_discrepancies(scores)

        # ------------------------------------------------------------------
        # 7. EXPLICACIÓN DE PESOS (para UI)
        # ------------------------------------------------------------------
        regime_detected = self._detect_regime(market_context)
        weight_explanation = self._explain_weights(
            regime_detected, dynamic_weights, ml_accuracy, available
        )

        return {
            'consensus_score':   round(consensus_score, 1),
            'confidence':        round(confidence, 1),
            'recommendation':    recommendation,
            'sources_used':      available,
            'source_scores':     scores,
            'discrepancies':     discrepancies,
            'weights_used':      {k: round(dynamic_weights[k], 1) for k in available},
            'regime_detected':   regime_detected,
            'weight_explanation': weight_explanation,
        }

    # =========================================================================
    # PESOS DINÁMICOS
    # =========================================================================

    def _compute_dynamic_weights(self,
                                  ml_prediction,
                                  lstm_prediction,
                                  groq_analysis,
                                  market_context,
                                  ml_accuracy) -> Dict[str, float]:
        """
        Calcula pesos ajustados por:
          A) Régimen de mercado (VIX)
          B) Calidad real del ML (accuracy post-leakage-fix)
          C) Disponibilidad de fuentes (redistribuye pesos faltantes)
        """
        # A) Base según régimen
        regime = self._detect_regime(market_context)
        weights = dict(REGIME_WEIGHTS[regime])   # copia mutable

        # B) Ajuste por calidad del ML
        #    Si accuracy real es conocida y está por debajo de 55%, reducimos
        #    su peso porque no tiene edge estadístico claro.
        if ml_accuracy is not None:
            if ml_accuracy < 0.52:
                # Sin edge — reducir ML y LSTM, dar más a Técnico
                excess = weights['ml'] * 0.5 + weights['lstm'] * 0.5
                weights['ml']        *= 0.5
                weights['lstm']      *= 0.5
                weights['technical'] += excess * 0.6
                weights['groq']      += excess * 0.4
            elif ml_accuracy < 0.55:
                # Edge débil — reducción moderada
                excess = weights['ml'] * 0.25 + weights['lstm'] * 0.25
                weights['ml']        *= 0.75
                weights['lstm']      *= 0.75
                weights['technical'] += excess * 0.7
                weights['groq']      += excess * 0.3
            elif ml_accuracy >= 0.60:
                # Edge sólido — aumentar peso de ML
                bonus = 5.0
                weights['ml']        += bonus
                weights['lstm']      += bonus
                weights['technical'] -= bonus
                weights['groq']      -= bonus

        # C) Redistribuir pesos de fuentes no disponibles
        missing = []
        if not (ml_prediction   and ml_prediction.get('probability_up')   is not None): missing.append('ml')
        if not (lstm_prediction  and lstm_prediction.get('probability_up') is not None): missing.append('lstm')
        if not groq_analysis: missing.append('groq')

        for src in missing:
            freed = weights[src]
            weights[src] = 0
            # El peso liberado va a técnico (principal fuente siempre disponible)
            weights['technical'] += freed * 0.6
            if 'groq' not in missing:
                weights['groq'] += freed * 0.4
            else:
                weights['technical'] += freed * 0.4

        # Normalizar a 100
        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] = (weights[k] / total) * 100

        return weights

    def _detect_regime(self, market_context: Optional[Dict]) -> str:
        """Detecta régimen según VIX o campo 'regime' del fetcher."""
        if not market_context:
            return 'NEUTRAL'

        # Si el fetcher ya nos da el régimen con su nomenclatura
        regime_str = market_context.get('regime', '').upper()
        if 'CRISIS' in regime_str:
            return 'CRISIS'
        if 'OFF' in regime_str:
            return 'RISK_OFF'
        if 'ON' in regime_str:
            return 'RISK_ON'

        # Fallback: calcular desde VIX numérico
        vix = market_context.get('vix', 20)
        for regime, (lo, hi) in VIX_THRESHOLDS.items():
            if lo <= vix < hi:
                return regime
        return 'NEUTRAL'

    def _explain_weights(self, regime: str, weights: Dict,
                          ml_accuracy: Optional[float],
                          available: List[str]) -> str:
        """Genera texto corto explicando por qué los pesos son los que son."""
        lines = [f"📐 Régimen detectado: **{regime}**"]

        if regime == 'RISK_ON':
            lines.append("• VIX bajo → ML y LSTM tienen mayor peso (mercado predecible)")
        elif regime == 'RISK_OFF':
            lines.append("• VIX elevado → Técnico y Groq pesan más (ML menos fiable en alta vol)")
        elif regime == 'CRISIS':
            lines.append("• VIX extremo → Técnico domina (señales cuantitativas más robustas)")

        if ml_accuracy is not None:
            if ml_accuracy < 0.52:
                lines.append(f"• ML accuracy {ml_accuracy*100:.1f}% < 52% → peso reducido 50% (sin edge estadístico)")
            elif ml_accuracy < 0.55:
                lines.append(f"• ML accuracy {ml_accuracy*100:.1f}% → peso reducido 25% (edge débil)")
            elif ml_accuracy >= 0.60:
                lines.append(f"• ML accuracy {ml_accuracy*100:.1f}% → peso aumentado (edge sólido ✅)")

        missing = [s for s in ['ml', 'lstm', 'groq'] if s not in available]
        if missing:
            lines.append(f"• {', '.join(missing).upper()} no disponible → peso redistribuido a Técnico")

        return "\n".join(lines)

    # =========================================================================
    # GROQ SENTIMENT
    # =========================================================================

    def _extract_groq_sentiment(self, groq_text: str) -> float:
        """Extrae score 0-100 del texto de Groq usando keywords ponderados."""
        # (keyword, peso_bullish)  — negativo = bearish
        weighted_keywords = [
            # Señales fuertes
            ('COMPRA FUERTE',         +30),
            ('COMPRA AGRESIVA',       +30),
            ('VENTA FUERTE',          -30),
            # Señales moderadas
            ('COMPRA MODERADA',       +20),
            ('COMPRA',                +15),
            ('VENTA',                 -15),
            # Momentum
            ('MOMENTUM ALCISTA',      +10),
            ('MOMENTUM BAJISTA',      -10),
            ('SESGO ALCISTA',         +10),
            ('SESGO BAJISTA',         -10),
            ('CONVERGENCIA ALCISTA',  +12),
            ('ALTA PROBABILIDAD',     +8),
            # Neutros/precaución
            ('ESPERAR',               -5),
            ('PRECAUCIÓN',            -5),
            ('MANTENER',               0),
            ('SIN OPERACIÓN',         -3),
        ]

        text_upper = groq_text.upper()
        raw_score = 0
        for kw, weight in weighted_keywords:
            if kw in text_upper:
                raw_score += weight

        # Convertir a 0-100 (raw va de ~-75 a +75)
        normalized = 50 + raw_score  # centro en 50
        return float(np.clip(normalized, 0, 100))

    # =========================================================================
    # CONFIANZA
    # =========================================================================

    def _calculate_confidence(self, scores: Dict, sources: List[str],
                               weights: Dict) -> float:
        """
        Confianza basada en:
          1. Número de fuentes
          2. Desviación estándar entre fuentes (penaliza divergencia)
          3. Peso de la fuente dominante (si una sola fuente domina, baja la confianza)
        """
        n = len(sources)
        if n == 0:
            return 0.0
        if n == 1:
            return 55.0

        base = {2: 60, 3: 70, 4: 80}.get(n, 60)

        # Penalización por divergencia
        vals = [scores[s] for s in sources]
        std  = float(np.std(vals))
        if std < 8:
            base += 15
        elif std < 15:
            base += 8
        elif std < 25:
            base += 0
        elif std < 35:
            base -= 10
        else:
            base -= 20

        # Penalización si una sola fuente tiene >60% del peso total
        total_w = sum(weights[s] for s in sources)
        if total_w > 0:
            max_share = max(weights[s] / total_w for s in sources)
            if max_share > 0.60:
                base -= 10   # demasiada dependencia de una sola fuente

        return float(np.clip(base, 0, 100))

    # =========================================================================
    # RECOMENDACIÓN
    # =========================================================================

    def _get_recommendation(self, score: float, confidence: float) -> str:
        if confidence >= 80:
            if score >= 70:   return "COMPRA FUERTE"
            if score >= 60:   return "COMPRA"
            if score <= 30:   return "VENTA FUERTE"
            if score <= 40:   return "VENTA"
            return "MANTENER"
        elif confidence >= 60:
            if score >= 75:   return "COMPRA FUERTE"
            if score >= 65:   return "COMPRA"
            if score <= 25:   return "VENTA FUERTE"
            if score <= 35:   return "VENTA"
            return "MANTENER"
        else:
            if score >= 80:   return "COMPRA (Baja Confianza)"
            if score <= 20:   return "VENTA (Baja Confianza)"
            return "ESPERAR — Señales Mixtas"

    # =========================================================================
    # DISCREPANCIAS
    # =========================================================================

    def _analyze_discrepancies(self, scores: Dict) -> List[str]:
        discrepancies = []
        items = list(scores.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                s1, v1 = items[i]
                s2, v2 = items[j]
                diff = abs(v1 - v2)
                if diff > 30:
                    d1 = "alcista" if v1 > 50 else "bajista"
                    d2 = "alcista" if v2 > 50 else "bajista"
                    discrepancies.append(
                        f"⚠️ {s1.upper()} ({d1}: {v1:.0f}) vs "
                        f"{s2.upper()} ({d2}: {v2:.0f}) — "
                        f"Diferencia de {diff:.0f} pts"
                    )
        return discrepancies

    # =========================================================================
    # FORMATO OUTPUT
    # =========================================================================

    def format_consensus_output(self, consensus: Dict, ticker: str) -> str:
        score      = consensus['consensus_score']
        confidence = consensus['confidence']
        rec        = consensus['recommendation']
        color      = "🟢" if "COMPRA" in rec else "🔴" if "VENTA" in rec else "🟡"

        lines = [
            f"## 🎯 Consensus Score — {ticker}",
            f"### Recomendación Final",
            f"# {color} **{rec}**",
            f"### Métricas",
            f"- **Consensus Score:** {score:.1f}/100",
            f"- **Confianza:** {confidence:.1f}%",
            f"- **Fuentes:** {len(consensus['sources_used'])}/4",
            f"- **Régimen:** {consensus.get('regime_detected', 'N/A')}",
            "",
            "### Breakdown por Fuente (Pesos Dinámicos)",
        ]

        emojis = {'technical': '📊', 'ml': '🤖', 'lstm': '🧠', 'groq': '💬'}
        for src, src_score in consensus['source_scores'].items():
            w = consensus['weights_used'][src]
            lines.append(f"- {emojis.get(src,'📈')} **{src.upper()}:** {src_score:.1f}/100  (Peso: {w:.1f}%)")

        if consensus.get('weight_explanation'):
            lines += ["", "### 📐 Por qué estos pesos", consensus['weight_explanation']]

        if consensus['discrepancies']:
            lines += ["", "### ⚠️ Señales Conflictivas"]
            lines += consensus['discrepancies']

        return "\n".join(lines)

    def _get_source_emoji(self, source: str) -> str:
        return {'technical': '📊', 'ml': '🤖', 'lstm': '🧠', 'groq': '💬'}.get(source, '📈')


# ============================================================================
# FUNCIÓN HELPER PARA STREAMLIT
# ============================================================================

def get_consensus_analysis(ticker:            str,
                            technical_analysis: Dict,
                            ml_prediction:      Optional[Dict] = None,
                            lstm_prediction:    Optional[Dict] = None,
                            groq_analysis:      Optional[str]  = None,
                            custom_weights:     Optional[Dict] = None,
                            market_context:     Optional[Dict] = None,
                            ml_accuracy:        Optional[float] = None) -> Dict:
    """
    Helper para Streamlit.  Acepta los mismos parámetros que antes +
    los nuevos opcionales (market_context, ml_accuracy).
    """
    analyzer = ConsensusAnalyzer(weights=custom_weights)
    technical_score = technical_analysis['signals']['score']

    consensus = analyzer.analyze_consensus(
        technical_score  = technical_score,
        ml_prediction    = ml_prediction,
        lstm_prediction  = lstm_prediction,
        groq_analysis    = groq_analysis,
        market_context   = market_context,
        ml_accuracy      = ml_accuracy,
    )
    consensus['ticker'] = ticker
    return consensus
