"""
CNIRS Crawler Middlewares Package

This package contains Scrapy middlewares for anti-detection and humanization.

Available Middlewares:
    - StealthMiddleware: Hides automation markers
    - PlaywrightStealthMiddleware: Enhanced stealth using playwright-stealth
    - HumanizationMiddleware: Simulates human browsing behavior
    - AdvancedHumanizationMiddleware: Advanced behavior patterns

Author: CNIRS Development Team
License: Educational Use Only
"""

from .stealth_middleware import StealthMiddleware, PlaywrightStealthMiddleware
from .humanization_middleware import HumanizationMiddleware, AdvancedHumanizationMiddleware

__all__ = [
    'StealthMiddleware',
    'PlaywrightStealthMiddleware',
    'HumanizationMiddleware',
    'AdvancedHumanizationMiddleware',
]
