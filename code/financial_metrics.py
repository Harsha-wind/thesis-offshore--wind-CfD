import numpy as np


def calculate_npv(cash_flows, discount_rate, initial_investment=0.0):
    """
    Calculate Net Present Value (NPV) of a series of cash flows.
    
    Parameters:
    -----------
    cash_flows : array-like
        Series of cash flows (revenue - opex) for each period
    discount_rate : float
        Discount rate per period (e.g., monthly WACC for monthly cash flows)
    initial_investment : float, optional
        Initial capital expenditure (CAPEX) at time 0. Default is 0.0
        This should be a positive number representing the investment cost.
    
    Returns:
    --------
    float
        Net Present Value in the same currency as cash_flows
    
    Example:
    --------
    >>> monthly_net_cash_flows = [10000, 12000, 15000]  # Revenue - OPEX
    >>> monthly_wacc = 0.004  # Monthly discount rate
    >>> capex = 500000  # Initial investment
    >>> npv = calculate_npv(monthly_net_cash_flows, monthly_wacc, capex)
    """
    # Convert to numpy array for calculation
    cash_flows = np.asarray(cash_flows, dtype=float)
    
    # Create period array: 1, 2, 3, ..., n
    periods = np.arange(1, len(cash_flows) + 1)
    
    # Calculate discount factors for each period: 1/(1+r)^t
    discount_factors = 1 / (1 + discount_rate) ** periods
    
    # Discount all future cash flows to present value
    discounted_flows = cash_flows * discount_factors
    
    # NPV = Sum of discounted cash flows - Initial Investment
    npv = discounted_flows.sum() - initial_investment
    
    return npv


def calculate_discounted_cash_flows(cash_flows, discount_rate):
    """
    Calculate the present value of each cash flow individually.
    
    Parameters:
    -----------
    cash_flows : array-like
        Series of cash flows for each period
    discount_rate : float
        Discount rate per period
    
    Returns:
    --------
    numpy.ndarray
        Array of discounted cash flows (present values)
    """
    cash_flows = np.asarray(cash_flows, dtype=float)
    periods = np.arange(1, len(cash_flows) + 1)
    discount_factors = 1 / (1 + discount_rate) ** periods
    
    return cash_flows * discount_factors

def calculate_debt_service(debt_amount, annual_interest_rate, loan_term_years, 
                           periods_per_year=12):
    """
    Calculate periodic debt service payment (principal + interest).
    
    Uses the annuity formula to calculate constant periodic payments.
    
    Parameters:
    -----------
    debt_amount : float
        Total loan amount (€)
    annual_interest_rate : float
        Annual interest rate (e.g., 0.045 for 4.5%)
    loan_term_years : int
        Loan duration in years
    periods_per_year : int, optional
        Number of payment periods per year (12 for monthly, 4 for quarterly)
        Default is 12 for monthly payments
    
    Returns:
    --------
    float
        Periodic debt service payment (€)
    
    Example:
    --------
    >>> debt = 1_800_000_000  # €1.8B loan
    >>> rate = 0.045  # 4.5% annual interest
    >>> term = 25  # 25-year loan
    >>> monthly_payment = calculate_debt_service(debt, rate, term, 12)
    """
    # Convert annual rate to periodic rate
    periodic_rate = annual_interest_rate / periods_per_year
    
    # Total number of payments
    total_periods = loan_term_years * periods_per_year
    
    # Annuity formula: PMT = P * [r(1+r)^n] / [(1+r)^n - 1]
    numerator = periodic_rate * (1 + periodic_rate) ** total_periods
    denominator = (1 + periodic_rate) ** total_periods - 1
    
    periodic_payment = debt_amount * (numerator / denominator)
    
    return periodic_payment


def calculate_dscr_metrics(net_cash_flows, debt_service_payment):
    """
    Calculate Debt Service Coverage Ratio (DSCR) metrics.
    
    DSCR measures the ability to cover debt obligations from operating cash flow.
    Banks typically require DSCR > 1.25-1.30 for project financing.
    
    Parameters:
    -----------
    net_cash_flows : array-like
        Net operating cash flows (Revenue - OPEX) for each period
    debt_service_payment : float
        Periodic debt service payment (principal + interest)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'dscr_values': Array of DSCR for each period
        - 'mean_dscr': Average DSCR over all periods
        - 'min_dscr': Minimum DSCR (worst case)
        - 'percentile_5_dscr': 5th percentile DSCR (CVaR equivalent)
        - 'prob_below_1_25': Probability of covenant breach (DSCR < 1.25)
        - 'prob_below_1_0': Probability of default (DSCR < 1.0)
        - 'months_below_1_25': Number of periods with DSCR < 1.25
    
    Example:
    --------
    >>> monthly_cf = [5_000_000, 4_500_000, 6_000_000, ...]
    >>> monthly_debt = 4_000_000
    >>> dscr_results = calculate_dscr_metrics(monthly_cf, monthly_debt)
    >>> print(f"Average DSCR: {dscr_results['mean_dscr']:.2f}")
    """
    net_cash_flows = np.asarray(net_cash_flows, dtype=float)
    
    # Calculate DSCR for each period
    dscr_values = net_cash_flows / debt_service_payment
    
    # Calculate key metrics
    mean_dscr = dscr_values.mean()
    min_dscr = dscr_values.min()
    percentile_5_dscr = np.percentile(dscr_values, 5)
    
    # Covenant breach analysis (typical bank requirement: DSCR ≥ 1.25)
    prob_below_1_25 = (dscr_values < 1.25).mean()
    months_below_1_25 = (dscr_values < 1.25).sum()
    
    # Default risk (DSCR < 1.0 means cannot cover debt)
    prob_below_1_0 = (dscr_values < 1.0).mean()
    
    return {
        'dscr_values': dscr_values,
        'mean_dscr': mean_dscr,
        'min_dscr': min_dscr,
        'percentile_5_dscr': percentile_5_dscr,
        'prob_below_1_25': prob_below_1_25,
        'prob_below_1_0': prob_below_1_0,
        'months_below_1_25': int(months_below_1_25)
    }