"""
Scrapy Settings for CNIRS News Crawlers

This module contains the default Scrapy settings for all news crawlers.

Author: CNIRS Development Team
License: Educational Use Only
"""

# Scrapy settings for CNIRS project

BOT_NAME = 'cnirs_crawler'

SPIDER_MODULES = ['scripts.crawlers']
NEWSPIDER_MODULE = 'scripts.crawlers'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 8

# Configure a delay for requests for the same website (default: 0)
# See https://docs.scrapy.org/en/latest/topics/settings.html#download-delay
DOWNLOAD_DELAY = 2
# The download delay setting will honor only one of:
CONCURRENT_REQUESTS_PER_DOMAIN = 2
#CONCURRENT_REQUESTS_PER_IP = 2

# Disable cookies (enabled by default)
COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
TELNETCONSOLE_ENABLED = False

# Override the default request headers:
DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
}

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
#SPIDER_MIDDLEWARES = {
#    'cnirs.middlewares.CnirsSpiderMiddleware': 543,
#}

# Enable or disable downloader middlewares
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
    'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
    # Playwright support
    'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler': 800,
    # Anti-detection middlewares
    'scripts.crawlers.middlewares.stealth_middleware.StealthMiddleware': 585,
    'scripts.crawlers.middlewares.humanization_middleware.HumanizationMiddleware': 586,
}

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    'scrapy.extensions.telnet.TelnetConsole': None,
#}

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
#ITEM_PIPELINES = {
#    'cnirs.pipelines.CnirsPipeline': 300,
#}

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
AUTOTHROTTLE_ENABLED = True
# The initial download delay
AUTOTHROTTLE_START_DELAY = 2
# The maximum download delay to be set in case of high latencies
AUTOTHROTTLE_MAX_DELAY = 10
# The average number of requests Scrapy should be sending in parallel to
# each remote server
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0
# Enable showing throttle stats for every response received:
#AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
#HTTPCACHE_ENABLED = True
#HTTPCACHE_EXPIRATION_SECS = 0
#HTTPCACHE_DIR = 'httpcache'
#HTTPCACHE_IGNORE_HTTP_CODES = []
#HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'

# Retry settings
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408, 429]

# User-Agent
USER_AGENT = 'CNIRS Academic Research Bot (contact: research@example.com; Educational Use Only)'

# Log settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
LOG_DATEFORMAT = '%Y-%m-%d %H:%M:%S'

# Feed settings (default output format)
FEEDS = {
    'data/raw/crawl_%(name)s_%(time)s.jsonl': {
        'format': 'jsonlines',
        'encoding': 'utf8',
        'store_empty': False,
        'overwrite': False,
        'indent': None,  # Compact JSON
    }
}

# Request fingerprinter implementation
REQUEST_FINGERPRINTER_IMPLEMENTATION = '2.7'

# Twisted reactor
#TWISTED_REACTOR = 'twisted.internet.asyncioreactor.AsyncioSelectorReactor'

# Feed export encoding
FEED_EXPORT_ENCODING = 'utf-8'

# DNS Timeout
DNS_TIMEOUT = 60

# Download timeout
DOWNLOAD_TIMEOUT = 60

# Media pipeline settings (for images, if needed)
# IMAGES_STORE = 'data/images'
# IMAGES_MIN_HEIGHT = 100
# IMAGES_MIN_WIDTH = 100

# ============================================================================
# Playwright Settings (Anti-Detection & Browser Automation)
# ============================================================================

# Enable Playwright for handling JavaScript-heavy pages
PLAYWRIGHT_BROWSER_TYPE = 'chromium'

# Launch options for Playwright browser
PLAYWRIGHT_LAUNCH_OPTIONS = {
    'headless': True,  # Run in headless mode (no GUI)
    'args': [
        '--disable-blink-features=AutomationControlled',  # Hide automation
        '--disable-dev-shm-usage',  # Overcome limited resource problems
        '--no-sandbox',  # Disable sandbox (for WSL/Linux compatibility)
        '--disable-setuid-sandbox',
        '--disable-gpu',  # Disable GPU acceleration
        '--disable-web-security',  # Disable web security (for CORS)
        '--disable-features=IsolateOrigins,site-per-process',
        '--window-size=1920,1080',  # Default window size
    ],
    'ignore_https_errors': True,
}

# Default navigation timeout (30 seconds)
PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 30000

# Maximum number of browser contexts (tabs)
PLAYWRIGHT_MAX_CONTEXTS = 4

# Maximum pages per context
PLAYWRIGHT_MAX_PAGES_PER_CONTEXT = 4

# Abort certain request types to speed up crawling
PLAYWRIGHT_ABORT_REQUEST = lambda req: req.resource_type in ['image', 'stylesheet', 'font']

# Process all request types through Playwright by default
PLAYWRIGHT_PROCESS_REQUEST_HEADERS = None

# Enable download handler for http and https
DOWNLOAD_HANDLERS = {
    'http': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
    'https': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
}

# ============================================================================
# Anti-Detection Middleware Settings
# ============================================================================

# Enable stealth mode to hide automation markers
PLAYWRIGHT_STEALTH_ENABLED = True

# Enable humanization behaviors
HUMANIZATION_ENABLED = True
HUMANIZATION_MIN_DELAY = 0.5  # Minimum delay between actions (seconds)
HUMANIZATION_MAX_DELAY = 2.0  # Maximum delay between actions (seconds)
HUMANIZATION_SCROLL_ENABLED = True  # Enable random scrolling
HUMANIZATION_READING_TIME = True  # Simulate reading time

# ============================================================================
# Cookies & Session Management
# ============================================================================

# Enable cookies for Playwright requests (session persistence)
COOKIES_ENABLED = True  # Override earlier setting for Playwright
COOKIES_DEBUG = False
