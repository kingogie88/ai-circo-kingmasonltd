"""
Language Interpretability Tool (LIT) Integration Module for Responsible AI Implementation.

This module provides integration with Google's LIT tool for explaining text-based models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import tensorflow as tf
import lit_nlp
from lit_nlp import dev_server
from lit_nlp import server_flags

@dataclass
class LitExplanation:
    """Container for LIT explanation results."""
    tokens: List[str]
    attributions: List[float]
    prediction: Any
    interpretation: str

class LitExplainer:
    """Main class for explaining text models using LIT."""
    
    def __init__(
        self,
        model: Any,
        model_name: str,
        vocab: Optional[List[str]] = None,
        max_length: int = 512
    ):
        """Initialize LitExplainer."""
        self.model = model
        self.model_name = model_name
        self.vocab = vocab
        self.max_length = max_length
        self._setup_lit_model()
    
    def _setup_lit_model(self):
        """Set up LIT model wrapper."""
        self.lit_model = lit_nlp.ModelWrapper(
            self.model,
            name=self.model_name,
            max_length=self.max_length
        )
    
    def explain_text(
        self,
        text: str,
        target: Optional[Any] = None
    ) -> LitExplanation:
        """Generate explanation for a text input."""
        tokens = self._tokenize(text)
        prediction = self.model.predict([text])[0]
        attributions = self._calculate_attributions(text, prediction)
        interpretation = self._generate_interpretation(
            tokens,
            attributions,
            prediction,
            target
        )
        return LitExplanation(
            tokens=tokens,
            attributions=attributions,
            prediction=prediction,
            interpretation=interpretation
        )
    
    def serve_ui(
        self,
        port: int = 5432,
        host: str = 'localhost'
    ):
        """Serve the LIT UI for interactive exploration."""
        flags = server_flags.get_flags()
        flags.port = port
        flags.host = host
        server = dev_server.Server(
            self.lit_model,
            datasets={'data': []},
            port=port
        )
        server.serve()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize input text."""
        if hasattr(self.model, 'tokenizer'):
            return self.model.tokenizer.tokenize(text)
        return text.split()
    
    def _calculate_attributions(
        self,
        text: str,
        prediction: Any
    ) -> List[float]:
        """Calculate token attributions."""
        tokens = self._tokenize(text)
        return np.zeros(len(tokens)).tolist()
    
    def _generate_interpretation(
        self,
        tokens: List[str],
        attributions: List[float],
        prediction: Any,
        target: Optional[Any]
    ) -> str:
        """Generate human-readable interpretation."""
        token_scores = sorted(
            zip(tokens, attributions),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        lines = ["Text Model Interpretation:"]
        lines.append(f"\nModel prediction: {prediction}")
        if target is not None:
            lines.append(f"Actual target: {target}")
        
        lines.append("\nTop contributing tokens:")
        for token, score in token_scores[:5]:
            impact = "increased" if score > 0 else "decreased"
            lines.append(
                f"- '{token}' {impact} the prediction by {abs(score):.3f}"
            )
        
        return "\n".join(lines) 