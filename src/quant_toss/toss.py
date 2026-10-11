"""Official Toss Invest v1.2.24 market-data adapter, deliberately read-only.

Reference checked 2026-10-11:
https://openapi.tossinvest.com/openapi-docs/latest/openapi.json
No account selection, order POST, credential issuance, or money movement.
"""
import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class TossMarketData:
    BASE = 'https://openapi.tossinvest.com'

    def __init__(self, token=None, transport=None):
        self._token = token or os.environ.get('TOSS_ACCESS_TOKEN')
        if not self._token:
            raise ValueError('Set TOSS_ACCESS_TOKEN locally; never paste credentials into chat or GitHub')
        self.transport = transport or urlopen

    def get(self, path, parameters):
        if path not in ('/api/v1/candles', '/api/v1/prices'):
            raise ValueError('Only read-only market-data endpoints are allowed')
        request = Request(self.BASE + path + '?' + urlencode(parameters),
                          headers={'Authorization': 'Bearer ' + self._token, 'Accept': 'application/json'},
                          method='GET')
        try:
            with self.transport(request, timeout=20) as response:
                payload = json.load(response)
        except HTTPError as error:
            # Do not echo server bodies, request headers, or credentials.
            raise RuntimeError(f'Toss HTTP {error.code}; check token, allowlisted IP and rate limits') from None
        except (URLError, TimeoutError):
            raise RuntimeError('Toss network request failed') from None
        if not isinstance(payload, dict) or 'result' not in payload:
            raise ValueError('Unexpected Toss response envelope')
        return payload['result']

    def daily_bars(self, symbol, pages=10):
        if not re.fullmatch(r'[A-Za-z0-9.\-]{1,20}', symbol) or not 1 <= pages <= 100:
            raise ValueError('Invalid symbol or page count')
        before, seen_cursors, rows, currency = None, set(), {}, None
        for _ in range(pages):
            query = {'symbol': symbol, 'interval': '1d', 'count': 200, 'adjusted': 'true'}
            if before:
                query['before'] = before
            page = self.get('/api/v1/candles', query)
            for candle in page['candles']:
                stamp = datetime.fromisoformat(candle['timestamp'])
                trading_date = stamp.date().isoformat()  # local date, do not shift KR bars to UTC
                # Exclude the current local date, which may be a partial daily bar.
                if stamp.tzinfo is None:
                    raise ValueError('Toss candle must include timezone offset')
                if trading_date >= datetime.now(stamp.tzinfo).date().isoformat():
                    continue
                if currency is not None and currency != candle['currency']:
                    raise ValueError('Mixed currency response')
                currency = candle['currency']
                row = {'date': trading_date, **{name: candle[key] for name, key in
                       [('open', 'openPrice'), ('high', 'highPrice'), ('low', 'lowPrice'),
                        ('close', 'closePrice'), ('volume', 'volume')]}}
                if trading_date in rows and rows[trading_date] != row:
                    raise ValueError('Conflicting duplicate daily bars')
                rows[trading_date] = row
            next_cursor = page.get('nextBefore')
            if not next_cursor or not page['candles']:
                break
            if next_cursor in seen_cursors:
                raise ValueError('Repeated pagination cursor')
            seen_cursors.add(next_cursor)
            before = next_cursor
            time.sleep(.25)
        if not rows:
            raise ValueError('No completed daily bars returned')
        return [rows[key] for key in sorted(rows)], currency

    def save_daily(self, symbol, output, pages=10):
        rows, currency = self.daily_bars(symbol, pages)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise ValueError('Refusing to overwrite existing market-data snapshot')
        with output.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['date', 'open', 'high', 'low', 'close', 'volume'])
            writer.writeheader()
            writer.writerows(rows)
        metadata = {'symbol': symbol, 'currency': currency, 'source': 'official Toss Invest API',
                    'api_version_checked': '1.2.24', 'adjusted': True, 'count': len(rows),
                    'fetched_at_utc': datetime.now(timezone.utc).isoformat(),
                    'note': 'Adjusted historical snapshot; verify corporate-action treatment before PnL claims'}
        output.with_suffix('.meta.json').write_text(json.dumps(metadata, indent=2))
        return metadata
