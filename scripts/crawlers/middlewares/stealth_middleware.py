#!/usr/bin/env python
"""
Stealth Middleware for Playwright Spiders

This middleware hides automation markers that websites use to detect bots:
- Overrides navigator.webdriver property
- Patches Chrome DevTools Protocol (CDP) runtime
- Randomizes WebGL vendor/renderer
- Spoofs permissions API
- Hides Playwright-specific properties

Usage:
    Add to DOWNLOADER_MIDDLEWARES in settings.py:
    'scripts.crawlers.middlewares.stealth_middleware.StealthMiddleware': 585

Author: CNIRS Development Team
License: Educational Use Only
"""

import logging
from typing import Union
from scrapy import signals
from scrapy.http import Request, Response
from scrapy.exceptions import NotConfigured

try:
    from scrapy_playwright.page import PageMethod
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

logger = logging.getLogger(__name__)


class StealthMiddleware:
    """
    Middleware to apply stealth techniques to Playwright requests.

    This middleware injects JavaScript code to hide automation markers
    that websites commonly check to detect bot traffic.

    Features:
        - Removes navigator.webdriver flag
        - Patches Chrome DevTools Protocol
        - Randomizes WebGL fingerprint
        - Spoofs permissions API
        - Disables automation-related properties
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize stealth middleware.

        Args:
            enabled: Whether to enable stealth mode
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise NotConfigured("scrapy-playwright is not installed")

        self.enabled = enabled
        if self.enabled:
            logger.info("StealthMiddleware enabled")

    @classmethod
    def from_crawler(cls, crawler):
        """Create middleware instance from crawler."""
        enabled = crawler.settings.getbool('PLAYWRIGHT_STEALTH_ENABLED', True)
        middleware = cls(enabled=enabled)

        crawler.signals.connect(
            middleware.spider_opened,
            signal=signals.spider_opened
        )

        return middleware

    def spider_opened(self, spider):
        """Log when spider opens."""
        logger.info(f"StealthMiddleware active for spider: {spider.name}")

    def process_request(self, request: Request, spider):
        """
        Process request to add stealth page methods.

        Args:
            request: Scrapy request object
            spider: Spider instance

        Returns:
            None (request is modified in place)
        """
        if not self.enabled:
            return None

        # Only process Playwright requests
        if not request.meta.get('playwright'):
            return None

        # Get existing page methods or create new list
        page_methods = request.meta.get('playwright_page_methods', [])
        if not isinstance(page_methods, list):
            page_methods = [page_methods]

        # Add stealth init script at the beginning
        stealth_init = PageMethod(
            'add_init_script',
            path=None,
            script=self._get_stealth_script()
        )

        # Insert at beginning to run before page load
        page_methods.insert(0, stealth_init)

        # Update request meta
        request.meta['playwright_page_methods'] = page_methods

        logger.debug(f"Applied stealth to request: {request.url}")

        return None

    def _get_stealth_script(self) -> str:
        """
        Get JavaScript code to hide automation markers.

        Returns:
            str: JavaScript code to inject

        Reference:
            Based on playwright-stealth and puppeteer-extra-plugin-stealth
        """
        return """
        // Override navigator.webdriver
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined,
        });

        // Override Chrome runtime
        window.chrome = {
            runtime: {},
        };

        // Override permissions API
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
        );

        // Override plugins
        Object.defineProperty(navigator, 'plugins', {
            get: () => [
                {
                    0: {type: "application/x-google-chrome-pdf", suffixes: "pdf", description: "Portable Document Format"},
                    description: "Portable Document Format",
                    filename: "internal-pdf-viewer",
                    length: 1,
                    name: "Chrome PDF Plugin"
                },
                {
                    0: {type: "application/pdf", suffixes: "pdf", description: ""},
                    description: "",
                    filename: "mhjfbmdgcfjbbpaeojofohoefgiehjai",
                    length: 1,
                    name: "Chrome PDF Viewer"
                },
                {
                    0: {type: "application/x-nacl", suffixes: "", description: "Native Client Executable"},
                    1: {type: "application/x-pnacl", suffixes: "", description: "Portable Native Client Executable"},
                    description: "",
                    filename: "internal-nacl-plugin",
                    length: 2,
                    name: "Native Client"
                }
            ],
        });

        // Override languages
        Object.defineProperty(navigator, 'languages', {
            get: () => ['zh-TW', 'zh', 'en-US', 'en'],
        });

        // Override WebGL vendor
        const getParameter = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.getParameter = function(parameter) {
            if (parameter === 37445) {
                return 'Intel Inc.';
            }
            if (parameter === 37446) {
                return 'Intel Iris OpenGL Engine';
            }
            return getParameter.apply(this, arguments);
        };

        // Override hairline feature detection
        if (!window.chrome) {
            Object.defineProperty(window, 'chrome', {
                get: () => ({
                    app: {
                        isInstalled: false,
                    },
                    webstore: {
                        onInstallStageChanged: {},
                        onDownloadProgress: {},
                    },
                    runtime: {
                        PlatformOs: {
                            MAC: 'mac',
                            WIN: 'win',
                            ANDROID: 'android',
                            CROS: 'cros',
                            LINUX: 'linux',
                            OPENBSD: 'openbsd',
                        },
                        PlatformArch: {
                            ARM: 'arm',
                            X86_32: 'x86-32',
                            X86_64: 'x86-64',
                        },
                        PlatformNaclArch: {
                            ARM: 'arm',
                            X86_32: 'x86-32',
                            X86_64: 'x86-64',
                        },
                        RequestUpdateCheckStatus: {
                            THROTTLED: 'throttled',
                            NO_UPDATE: 'no_update',
                            UPDATE_AVAILABLE: 'update_available',
                        },
                        OnInstalledReason: {
                            INSTALL: 'install',
                            UPDATE: 'update',
                            CHROME_UPDATE: 'chrome_update',
                            SHARED_MODULE_UPDATE: 'shared_module_update',
                        },
                        OnRestartRequiredReason: {
                            APP_UPDATE: 'app_update',
                            OS_UPDATE: 'os_update',
                            PERIODIC: 'periodic',
                        },
                    },
                }),
            });
        }

        // Modernizr hairline fix
        if (!window.Modernizr) {
            Object.defineProperty(window, 'Modernizr', {
                get: () => ({}),
            });
        }

        // Remove Playwright-specific properties
        delete navigator.__proto__.webdriver;

        // Override toString methods to prevent detection
        const originalToString = Function.prototype.toString;
        Function.prototype.toString = function() {
            if (this === navigator.permissions.query) {
                return 'function query() { [native code] }';
            }
            if (this === WebGLRenderingContext.prototype.getParameter) {
                return 'function getParameter() { [native code] }';
            }
            return originalToString.apply(this, arguments);
        };

        console.log('[Stealth] Anti-detection script loaded');
        """


class PlaywrightStealthMiddleware(StealthMiddleware):
    """
    Enhanced stealth middleware using playwright-stealth package.

    This version uses the playwright-stealth library for more comprehensive
    anti-detection coverage.

    Note:
        Requires: pip install playwright-stealth
    """

    def __init__(self, enabled: bool = True):
        """Initialize enhanced stealth middleware."""
        super().__init__(enabled=enabled)

        try:
            from playwright_stealth import stealth_async
            self.stealth_async = stealth_async
            self.use_stealth_lib = True
            logger.info("Using playwright-stealth library for enhanced protection")
        except ImportError:
            self.use_stealth_lib = False
            logger.warning(
                "playwright-stealth not available, using basic stealth script. "
                "Install with: pip install playwright-stealth"
            )

    async def apply_stealth_to_page(self, page):
        """
        Apply stealth to Playwright page using library.

        Args:
            page: Playwright page object
        """
        if self.use_stealth_lib:
            try:
                await self.stealth_async(page)
                logger.debug("Applied playwright-stealth to page")
            except Exception as e:
                logger.error(f"Failed to apply playwright-stealth: {e}")
