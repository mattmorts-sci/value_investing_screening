"""
Yahoo Finance integration for fetching current market data.

Includes currency anomaly detection and ticker mapping support.
"""
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import yfinance as yf

from config.user_config import AnalysisConfig
from data_pipeline.ticker_mapping import TickerMapper

logger = logging.getLogger(__name__)


class YahooData:
    """Container for Yahoo Finance data."""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.current_price: Optional[float] = None
        self.current_market_cap: Optional[float] = None
        self.current_shares: Optional[float] = None
        self.current_fcf: Optional[float] = None
        self.current_operating_income: Optional[float] = None
        self.current_lt_debt: Optional[float] = None
        self.current_cash: Optional[float] = None
        self.currency: Optional[str] = None
        self.financial_currency: Optional[str] = None  # Currency of financial statements
        self.api_success: bool = False
        self.error_message: Optional[str] = None
        self.anomalies: Dict[str, str] = {}


def convert_ticker_format(ticker: str, country: str, mapper: Optional[TickerMapper] = None) -> str:
    """
    Convert internal ticker format to Yahoo Finance format.
    
    Args:
        ticker: Internal format (SYMBOL:COUNTRY or just SYMBOL)
        country: Default country if not in ticker
        mapper: Optional ticker mapper for handling changed symbols
        
    Returns:
        Yahoo Finance formatted ticker
    """
    # Apply mapping if available
    if mapper:
        ticker = mapper.get_mapped_ticker(ticker)
    
    # Parse ticker
    if ':' in ticker:
        symbol, ticker_country = ticker.split(':', 1)
    else:
        symbol = ticker
        ticker_country = country
    
    # Convert to Yahoo format
    if ticker_country == 'AU':
        return f"{symbol}.AX"
    elif ticker_country == 'US':
        return symbol
    else:
        raise ValueError(f"Unsupported country: {ticker_country}")


def fetch_current_prices(tickers: List[str], config: AnalysisConfig) -> Dict[str, YahooData]:
    """
    Fetch current price data from Yahoo Finance.
    
    Uses caching to minimize API calls.
    Includes currency anomaly detection.
    
    Args:
        tickers: List of tickers in internal format
        config: Analysis configuration
        
    Returns:
        Dictionary mapping ticker to YahooData
    """
    # Initialize ticker mapper if enabled
    mapper = TickerMapper() if config.ENABLE_TICKER_MAPPING else None
    
    # Check cache first
    cache_file = config.CACHE_DIRECTORY / f"yahoo_cache_{config.COUNTRY}.json"
    cached_data = _load_cache(cache_file, config.CACHE_HOURS)
    
    results = {}
    tickers_to_fetch = []
    
    # Use cached data where available
    for ticker in tickers:
        if cached_data and ticker in cached_data:
            data = YahooData(ticker)
            data.__dict__.update(cached_data[ticker])
            results[ticker] = data
        else:
            tickers_to_fetch.append(ticker)
    
    # Fetch missing data
    if tickers_to_fetch:
        logger.info(f"Fetching data for {len(tickers_to_fetch)} companies from Yahoo Finance")
        
        for ticker in tickers_to_fetch:
            data = _fetch_single_ticker(ticker, config, mapper)
            results[ticker] = data
            
            # Rate limiting
            time.sleep(0.1)
    
    # Update cache
    _save_cache(results, cache_file)
    
    # Summary
    successful = sum(1 for data in results.values() if data.api_success)
    with_anomalies = sum(1 for data in results.values() if data.anomalies)
    
    logger.info(f"Successfully fetched data for {successful}/{len(tickers)} companies")
    if with_anomalies:
        logger.warning(f"{with_anomalies} companies have data anomalies")
    
    return results

def _fetch_single_ticker(ticker: str, config: AnalysisConfig, mapper: Optional[TickerMapper]) -> YahooData:
    """Fetch data for a single ticker with retries."""
    data = YahooData(ticker)
    
    # Expected currency based on country
    expected_currency = 'AUD' if config.COUNTRY == 'AU' else 'USD'
    
    # Convert ticker format
    try:
        yahoo_ticker = convert_ticker_format(ticker, config.COUNTRY, mapper)
    except ValueError as e:
        data.error_message = str(e)
        return data
    
    # Retry logic
    for attempt in range(config.API_RETRY_ATTEMPTS):
        try:
            # Fetch data
            stock = yf.Ticker(yahoo_ticker)
            info = stock.info
            
            # Extract required fields from info dict
            data.current_price = (
                info.get('currentPrice') or 
                info.get('regularMarketPrice') or 
                info.get('previousClose')
            )
            
            data.current_market_cap = info.get('marketCap')
            data.current_shares = (
                info.get('sharesOutstanding') or 
                info.get('impliedSharesOutstanding')
            )
            
            data.current_fcf = (
                info.get('freeCashflow') or 
                info.get('free_cash_flow') or 
                info.get('freeCashFlow')
            )
            
            # Get operating income from income statement
            try:
                income_stmt = stock.income_stmt
                if income_stmt is not None and not income_stmt.empty:
                    # Log available rows for debugging
                    logger.debug(f"{ticker}: Income statement rows: {list(income_stmt.index)[:20]}...")
                    
                    # Try different possible row names
                    operating_income_rows = [
                        "Operating Income",
                        "EBIT",
                        "Earnings Before Interest and Tax",
                        "Operating Profit",
                        "Total Operating Income",
                        "Income From Operations",
                        "Operating Income (Loss)"
                    ]
                    
                    for row_name in operating_income_rows:
                        if row_name in income_stmt.index:
                            # Get the most recent value (first column)
                            data.current_operating_income = income_stmt.loc[row_name].iloc[0]
                            logger.debug(f"{ticker}: Found operating income as '{row_name}' = {data.current_operating_income}")
                            break
                    
                    if not data.current_operating_income:
                        logger.debug(f"{ticker}: Operating income not found in income statement")
                else:
                    logger.debug(f"{ticker}: Income statement is empty or None")
                    
            except Exception as e:
                logger.debug(f"{ticker}: Error fetching income statement: {e}")
                # Fall back to info dict if available
                data.current_operating_income = (
                    info.get('operatingIncome') or 
                    info.get('operating_income') or
                    info.get('ebit') or
                    info.get('EBIT')
                )
            
            data.current_lt_debt = (
                info.get('totalDebt') or 
                info.get('longTermDebt') or 
                info.get('total_debt')
            )
            
            data.current_cash = (
                info.get('totalCash') or 
                info.get('cash') or 
                info.get('totalCashAndCashEquivalents')
            )
            
            data.currency = info.get('currency')
            data.financial_currency = info.get('financialCurrency')
            
            # Validate
            if data.current_price and data.current_price > 0:
                data.api_success = True
                
                # Check for anomalies
                data.anomalies = validate_yahoo_data(data, expected_currency)
                
                return data
            else:
                data.error_message = "Invalid price data"
                
        except Exception as e:
            data.error_message = str(e)
            logger.debug(f"Attempt {attempt + 1} failed for {ticker}: {e}")
            
            if attempt < config.API_RETRY_ATTEMPTS - 1:
                time.sleep(config.API_RETRY_DELAY)
    
    return data



def validate_yahoo_data(data: YahooData, expected_currency: str) -> Dict[str, Any]:
    """
    Validate Yahoo Finance data for anomalies.
    
    Args:
        data: Yahoo Finance data
        expected_currency: Expected currency code
        
    Returns:
        Dictionary of anomalies found
    """
    anomalies = {}
    
    # Currency check
    if data.currency and data.currency != expected_currency:
        anomalies['currency_mismatch'] = f"Expected {expected_currency}, got {data.currency}"
    
    # Financial vs trading currency mismatch
    if data.financial_currency and data.currency and data.financial_currency != data.currency:
        anomalies['currency_reporting_mismatch'] = (
            f"Trading: {data.currency}, Reporting: {data.financial_currency}"
        )
    
    # Extreme FCF yield check
    if all([data.current_fcf, data.current_market_cap, 
            data.current_fcf > 0, data.current_market_cap > 0]):
        fcf_yield = data.current_fcf / data.current_market_cap
        if fcf_yield > 0.3:  # 30% FCF yield is suspicious
            anomalies['extreme_fcf_yield'] = f"{fcf_yield:.2%}"
    
    # Negative metrics that should be positive
    if data.current_shares and data.current_shares < 0:
        anomalies['negative_shares'] = f"{data.current_shares:,.0f}"
    
    if data.current_market_cap and data.current_market_cap < 0:
        anomalies['negative_market_cap'] = f"{data.current_market_cap:,.0f}"
    
    # Data consistency checks
    if all([data.current_price, data.current_shares, data.current_market_cap]):
        implied_market_cap = data.current_price * data.current_shares
        actual_market_cap = data.current_market_cap
        
        if actual_market_cap > 0:
            discrepancy = abs(implied_market_cap - actual_market_cap) / actual_market_cap
            if discrepancy > 0.2:  # 20% discrepancy
                anomalies['market_cap_inconsistency'] = (
                    f"Implied: {implied_market_cap:,.0f}, "
                    f"Actual: {actual_market_cap:,.0f} "
                    f"({discrepancy:.1%} difference)"
                )
    
    return anomalies


def _load_cache(cache_file: Path, max_age_hours: int) -> Optional[Dict]:
    """Load cached data if fresh enough."""
    if not cache_file.exists():
        return None
    
    # Check age
    file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
    if file_age > timedelta(hours=max_age_hours):
        logger.info("Cache expired, will fetch fresh data")
        return None
    
    # Load data
    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading cache: {e}")
        return None


def _save_cache(results: Dict[str, YahooData], cache_file: Path):
    """Save results to cache."""
    # Ensure directory exists
    cache_file.parent.mkdir(exist_ok=True)
    
    # Convert to serializable format
    cache_data = {}
    for ticker, data in results.items():
        if data.api_success:  # Only cache successful fetches
            cache_data[ticker] = {
                'current_price': data.current_price,
                'current_market_cap': data.current_market_cap,
                'current_shares': data.current_shares,
                'current_fcf': data.current_fcf,
                'current_operating_income': data.current_operating_income,
                'current_lt_debt': data.current_lt_debt,
                'current_cash': data.current_cash,
                'currency': data.currency,
                'financial_currency': data.financial_currency,
                'api_success': data.api_success,
                'anomalies': data.anomalies
            }
    
    # Save
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        logger.debug(f"Saved {len(cache_data)} entries to cache")
    except Exception as e:
        logger.warning(f"Error saving cache: {e}")


# def calculate_current_metrics(data: YahooData) -> Dict[str, float]:
#     """
#     Calculate additional metrics from Yahoo Finance data.
    
#     Args:
#         data: Yahoo Finance data
        
#     Returns:
#         Dictionary of calculated metrics
#     """
#     metrics = {}
    
#     # Calculate enterprise value if we have the components
#     if all([data.current_market_cap, data.current_lt_debt, data.current_cash]):
#         metrics['current_enterprise_value'] = (
#             data.current_market_cap + 
#             (data.current_lt_debt or 0) - 
#             (data.current_cash or 0)
#         )
        
#         # Calculate acquirer's multiple if we have operating income
#         if data.current_operating_income and data.current_operating_income > 0:
#             metrics['current_acquirers_multiple'] = (
#                 metrics['current_enterprise_value'] / 
#                 data.current_operating_income
#             )
    
#     # Calculate debt to cash ratio
#     if data.current_cash and data.current_cash > 0:
#         metrics['current_debt_cash_ratio'] = (data.current_lt_debt or 0) / data.current_cash
    
#     # FCF yield
#     if data.current_fcf and data.current_market_cap and data.current_market_cap > 0:
#         metrics['current_fcf_yield'] = data.current_fcf / data.current_market_cap
    
#     return metrics


def calculate_current_metrics(data: YahooData) -> Dict[str, float]:
    """
    Calculate additional metrics from Yahoo Finance data.
    
    Args:
        data: Yahoo Finance data
        
    Returns:
        Dictionary of calculated metrics
    """
    metrics = {}
    
    # DEBUG: Log what data we have
    logger.debug(f"{data.ticker}: Calculating metrics with market_cap={data.current_market_cap}, "
                f"debt={data.current_lt_debt}, cash={data.current_cash}, "
                f"operating_income={data.current_operating_income}")
    
    # Calculate enterprise value if we have the components
    if all([data.current_market_cap, data.current_lt_debt, data.current_cash]):
        metrics['current_enterprise_value'] = (
            data.current_market_cap + 
            (data.current_lt_debt or 0) - 
            (data.current_cash or 0)
        )
        
        # Calculate acquirer's multiple if we have operating income
        if data.current_operating_income and data.current_operating_income > 0:
            metrics['current_acquirers_multiple'] = (
                metrics['current_enterprise_value'] / 
                data.current_operating_income
            )
            logger.debug(f"{data.ticker}: Calculated AM = {metrics['current_acquirers_multiple']:.2f}")
        else:
            logger.debug(f"{data.ticker}: Cannot calculate AM - operating income = {data.current_operating_income}")
    else:
        logger.debug(f"{data.ticker}: Cannot calculate EV - missing components")
    
    # Calculate debt to cash ratio
    if data.current_cash and data.current_cash > 0:
        metrics['current_debt_cash_ratio'] = (data.current_lt_debt or 0) / data.current_cash
    
    # FCF yield
    if data.current_fcf and data.current_market_cap and data.current_market_cap > 0:
        metrics['current_fcf_yield'] = data.current_fcf / data.current_market_cap
    
    return metrics