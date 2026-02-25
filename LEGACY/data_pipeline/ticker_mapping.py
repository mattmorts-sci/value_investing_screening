"""
Ticker mapping system for handling changed company symbols.

Manages mappings for companies that have changed their ticker symbols
due to mergers, acquisitions, or rebranding.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TickerMapper:
    """Manages ticker symbol mappings for companies that changed symbols."""
    
    def __init__(self, mapping_file: Path = Path("cache/ticker_mappings.json")):
        """
        Initialize ticker mapper.
        
        Args:
            mapping_file: Path to JSON file storing mappings
        """
        self.mapping_file = mapping_file
        self.mappings = self._load_mappings()
        self._reverse_mappings = {v: k for k, v in self.mappings.items()}
    
    def _load_mappings(self) -> Dict[str, str]:
        """Load mappings from file."""
        if not self.mapping_file.exists():
            logger.info("No ticker mappings file found, starting with empty mappings")
            return {}
        
        try:
            with open(self.mapping_file, 'r') as f:
                mappings = json.load(f)
                logger.info(f"Loaded {len(mappings)} ticker mappings")
                return mappings
        except Exception as e:
            logger.warning(f"Error loading ticker mappings: {e}")
            return {}
    
    def _save_mappings(self):
        """Save mappings to file."""
        self.mapping_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(self.mapping_file, 'w') as f:
                json.dump(self.mappings, f, indent=2, sort_keys=True)
            logger.debug(f"Saved {len(self.mappings)} ticker mappings")
        except Exception as e:
            logger.error(f"Error saving ticker mappings: {e}")
    
    def add_mapping(self, old_ticker: str, new_ticker: str):
        """
        Add a ticker mapping.
        
        Args:
            old_ticker: Previous ticker symbol
            new_ticker: Current ticker symbol
        """
        if old_ticker in self.mappings:
            logger.warning(f"Overwriting existing mapping for {old_ticker}")
        
        self.mappings[old_ticker] = new_ticker
        self._reverse_mappings[new_ticker] = old_ticker
        self._save_mappings()
        logger.info(f"Added ticker mapping: {old_ticker} -> {new_ticker}")
    
    def get_mapped_ticker(self, ticker: str) -> str:
        """
        Get mapped ticker if exists, otherwise return original.
        
        Args:
            ticker: Ticker to look up
            
        Returns:
            Current ticker symbol
        """
        return self.mappings.get(ticker, ticker)
    
    def get_original_ticker(self, ticker: str) -> str:
        """
        Get original ticker if this is a mapped ticker.
        
        Args:
            ticker: Current ticker symbol
            
        Returns:
            Original ticker symbol if mapped, otherwise input ticker
        """
        return self._reverse_mappings.get(ticker, ticker)
    
    def remove_mapping(self, ticker: str):
        """
        Remove a ticker mapping.
        
        Args:
            ticker: Old ticker to remove mapping for
        """
        if ticker in self.mappings:
            new_ticker = self.mappings[ticker]
            del self.mappings[ticker]
            if new_ticker in self._reverse_mappings:
                del self._reverse_mappings[new_ticker]
            self._save_mappings()
            logger.info(f"Removed ticker mapping for {ticker}")
        else:
            logger.warning(f"No mapping found for {ticker}")
    
    def bulk_add_mappings(self, mappings: Dict[str, str]):
        """
        Add multiple mappings at once.
        
        Args:
            mappings: Dictionary of old_ticker -> new_ticker
        """
        for old_ticker, new_ticker in mappings.items():
            self.mappings[old_ticker] = new_ticker
            self._reverse_mappings[new_ticker] = old_ticker
        
        self._save_mappings()
        logger.info(f"Added {len(mappings)} ticker mappings")
    
    def get_all_mappings(self) -> Dict[str, str]:
        """Get all current mappings."""
        return self.mappings.copy()
    
    def clear_mappings(self):
        """Clear all mappings."""
        self.mappings.clear()
        self._reverse_mappings.clear()
        self._save_mappings()
        logger.info("Cleared all ticker mappings")