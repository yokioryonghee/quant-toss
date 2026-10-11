"""Standalone, offline HTML report for completed comparison demos."""
import argparse
import csv
import html
import json
import math
from pathlib import Path


def number(value, digits=3):
    if value is None:
        return '—'
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Report values must be finite')
    return f'{value:,.{digits}f}'


def percent(value):
    return number(float(value) * 100, 2) + '%'


def chart(series, *, suffix='', labels=None):
    """Render actual values as accessible SVG; do not rescale returns to look profitable."""
    values = [float(v) for _, points, _ in series for v in points]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError('Chart requires finite observations')
    low, high = min(values), max(values)
    padding = (high - low) * .12 or max(abs(high) * .02, .001)
    low, high = low - padding, high + padding
    out = ['<svg viewBox="0 0 700 240" role="img" aria-label="시계열 결과 차트">']
    for i in range(5):
        y = 16 + i * 46
        value = high - (high - low) * i / 4
        out.append(f'<line x1="74" y1="{y}" x2="680" y2="{y}" stroke="#253649"/>')
        out.append(f'<text x="66" y="{y+4}" text-anchor="end">{number(value, 2)}{html.escape(suffix)}</text>')
    for name, points, color in series:
        coords = ' '.join(f'{74+606*i/max(len(points)-1,1):.2f},{16+184*(high-float(v))/(high-low):.2f}'
                          for i, v in enumerate(points))
        out.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{coords}"><title>{html.escape(name)}</title></polyline>')
    if labels:
        for x, text, anchor in ((74, labels[0], 'start'), (680, labels[-1], 'end')):
            out.append(f'<text x="{x}" y="226" text-anchor="{anchor}">{html.escape(str(text))}</text>')
    out.append('</svg><div class="legend">')
    for name, _, color in series:
        out.append(f'<span><i style="background:{color}"></i>{html.escape(name)}</span>')
    return ''.join(out) + '</div>'


def render(folder):
    folder = Path(folder).resolve()
    read = lambda path: json.loads((folder / path).read_text(encoding='utf-8'))
    forecast = read('forecast.json')
    reports = {name: read(f'{name}/report.json') for name in ('linear', 'transformer')}
    histories = {name: read(f'{name}/history.json') for name in reports}
    ledger = read('transformer/ledger.json')
    if not ledger:
        raise ValueError('Completed demo ledger is empty')
    with (folder / 'synthetic.csv').open(encoding='utf-8', newline='') as handle:
        bars = list(csv.DictReader(handle))
    if not bars:
        raise ValueError('Completed demo price data is empty')
    synthetic = forecast.get('synthetic') is True
    if any(r.get('synthetic') is not synthetic for r in reports.values()):
        raise ValueError('Inconsistent synthetic-data labels')
    symbol = html.escape(str(forecast['symbol']))
    signal_date = html.escape(str(forecast['signal_date']))
    strategy = reports['transformer']['strategy']
    badge = 'SYNTHETIC · OFFLINE DEMO' if synthetic else 'CSV RESEARCH · PAPER ONLY'
    notice = ('합성 데이터로 실행한 결과입니다. 실제 주가·실제 수익·실계좌 운영 성과가 아닙니다.'
              if synthetic else 'CSV 기반 연구 결과입니다. 실제 주문이나 실계좌 체결 결과가 아닙니다.')
    cards = ''.join(f'<div class="stat"><span>{name}</span><strong>{value}</strong></div>' for name, value in (
        ('입력 거래일', str(len(bars))), ('모의매매 세션', str(strategy['session_count'])),
        ('거래한 세션', str(strategy['traded_sessions'])), ('비용 차감 모의 수익률', percent(strategy['total_return']))))
    ranges = ''.join(f'<tr><td>{html.escape(str(h))} 거래일</td><td>{percent(q["q10"])}</td>'
                     f'<td class="accent">{percent(q["median"])}</td><td>{percent(q["q90"])}</td></tr>'
                     for h, q in forecast['return_quantiles_from_next_open'].items())
    comparisons = ''
    for name, report in reports.items():
        metric = report['forecast_calibrated']['1']
        comparisons += (f'<tr><td>{html.escape(name.title())}</td><td>{number(metric["pinball"])}</td>'
                        f'<td>{number(metric["mae_percentage_points"])}</td><td>{percent(metric["interval_coverage"])}</td>'
                        f'<td>{percent(report["strategy"]["total_return"])}</td></tr>')
    empirical = reports['transformer']['empirical_train_quantiles']['1']
    comparisons += (f'<tr><td>Train empirical quantiles</td><td>{number(empirical["pinball"])}</td>'
                    f'<td>{number(empirical["mae_percentage_points"])}</td><td>{percent(empirical["interval_coverage"])}</td>'
                    f'<td>{percent(reports["transformer"]["empirical_strategy"]["total_return"])}</td></tr>')
    prices = chart([('종가 · 마지막 160개 세션', [float(b['close']) for b in bars[-160:]], '#65d6cb')],
                   labels=[bars[-min(160, len(bars))]['date'], bars[-1]['date']])
    losses = chart([(name.title() + ' · validation', [row['val_pinball'] for row in rows], color)
                    for (name, rows), color in zip(histories.items(), ('#a78bfa', '#65d6cb'))],
                   labels=['Epoch 1', 'Epoch ' + str(max(len(rows) for rows in histories.values()))])
    equity = chart([('Transformer · 누적 모의 수익률', [(row['equity']-1)*100 for row in ledger], '#65d6cb')],
                   suffix='%', labels=[ledger[0]['execution_date'], ledger[-1]['execution_date']])
    cash_note = ('조건을 충족한 거래가 없어 현금 보유를 유지했습니다. 0%는 프로그램 오류나 수익 달성의 증거가 아닙니다.'
                 if strategy['traded_sessions'] == 0 else '시가 진입·당일 종가 청산, 비용 차감 모의매매 기록입니다.')
    splits = ''.join(f'<tr><td>{html.escape(name)}</td><td>{row["count"]}</td>'
                     f'<td>{html.escape(row["first_signal"])}</td><td>{html.escape(row["last_label"])}</td></tr>'
                     for name, row in reports['transformer']['split_ranges'].items())
    document = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quant Toss · 실행 결과</title><style>
:root{{color-scheme:dark;font-family:system-ui,-apple-system,sans-serif;background:#0b121b;color:#e8eef6}}
*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1160px;margin:auto;padding:42px 24px 60px}}
.brand{{color:#65d6cb;font-weight:750;letter-spacing:2px}}h1{{font-size:clamp(28px,4vw,44px);margin:16px 0 10px}}h2{{font-size:20px;margin:0 0 16px}}
p{{color:#a7b5c8;line-height:1.75;margin:10px 0}}.badge{{display:inline-block;border:1px solid #3c716c;color:#65d6cb;border-radius:30px;padding:7px 12px;font-size:12px;margin:18px 0}}
.notice{{border-left:3px solid #eab761;padding:12px 16px;background:#201c18;border-radius:4px;color:#ebd8b8}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}}.stat,.panel{{background:#131e2b;border:1px solid #253649;border-radius:14px;padding:22px}}
.stat span{{display:block;color:#a7b5c8;font-size:13px}}.stat strong{{display:block;font-size:29px;margin-top:12px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}.panel{{margin-bottom:20px;min-width:0}}.wide{{grid-column:1/-1}}.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;white-space:nowrap;font-size:14px}}th,td{{padding:13px 10px;text-align:left;border-bottom:1px solid #253649}}th{{color:#a7b5c8;font-weight:500}}
.accent,a{{color:#65d6cb}}a{{text-decoration:none}}a:hover{{text-decoration:underline}}svg{{width:100%;height:auto}}svg text{{font-size:12px;fill:#a7b5c8}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:#a7b5c8;margin:6px 0}}.legend i{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}}
.links{{display:flex;gap:20px;flex-wrap:wrap;margin-top:18px}}footer{{color:#8193ac;font-size:13px;line-height:1.8}}
@media(max-width:760px){{.stats{{grid-template-columns:repeat(2,1fr)}}.grid{{grid-template-columns:1fr}}main{{padding:24px 16px}}.stat strong{{font-size:24px}}}}
</style></head><body><main>
<div class="brand">QUANT TOSS / RESEARCH</div><h1>실행 완료 · 결과 보고서</h1>
<p>{symbol} · {html.escape(str(forecast['currency']))} · 마지막 관측 {signal_date}</p>
<span class="badge">{badge}</span><div class="notice">{notice} 주문은 전송하지 않았습니다.</div>
<div class="stats">{cards}</div><div class="grid">
<section class="panel"><h2>입력 가격 흐름</h2>{prices}<p>학습에 사용한 CSV의 종가입니다. 예측 가격이 아닙니다.</p></section>
<section class="panel"><h2>학습 진행 · validation pinball</h2>{losses}<p>오차는 낮을수록 좋습니다. 데모는 각 모델을 2 epochs 학습합니다.</p></section>
<section class="panel"><h2>다음 시가 대비 수익률 예측</h2><div class="scroll"><table><thead><tr><th>기간</th><th>q10</th><th>중앙값</th><th>q90</th></tr></thead><tbody>{ranges}</tbody></table></div>
<p>{signal_date}까지의 관측으로 계산했습니다. 범위는 보정된 q10–q90이며 미래 수익이나 포함 확률을 보장하지 않습니다.</p></section>
<section class="panel"><h2>비용 차감 모의매매</h2>{equity}<p>{cash_note}</p></section>
<section class="panel wide"><h2>미래 구간 평가 · 1 거래일 head</h2><div class="scroll"><table><thead><tr><th>모델</th><th>Pinball ↓</th><th>MAE (퍼센트포인트) ↓</th><th>구간 포함률</th><th>모의 수익률</th></tr></thead><tbody>{comparisons}</tbody></table></div>
<p>동일한 시간 분리 기준의 test 결과입니다. 합성 데이터 비교로 실제 시장에서의 모델 우위를 주장하지 않습니다.</p></section>
<section class="panel wide"><h2>시간 순서와 미래 라벨 분리</h2><div class="scroll"><table><thead><tr><th>구간</th><th>샘플</th><th>첫 관측 종료일</th><th>마지막 target 날짜</th></tr></thead><tbody>{splits}</tbody></table></div>
<p>Train-only scaling · label purge · embargo · 독립 calibration. 5·20 거래일 head는 예측 평가용이고 매매에는 1 거래일 head만 사용합니다.</p>
<div class="links"><a href="forecast.json">예측 JSON</a><a href="transformer/report.json">평가 JSON</a><a href="transformer/ledger.json">모의매매 기록</a></div></section>
</div><footer>완료된 실행 결과를 표시하는 정적 보고서입니다. 자동 학습이나 실시간 시세 갱신을 수행하지 않습니다.<br>
로컬 파일로 열 수 있으며 외부 스크립트·네트워크 요청이 없습니다.</footer></main></body></html>'''
    output = folder / 'report.html'
    output.write_text(document, encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description='Generate an offline HTML report from a completed demo')
    parser.add_argument('--input', required=True, help='Demo output folder containing forecast.json and both models')
    parser.add_argument('--open', action='store_true', help='Open the completed report in your browser')
    args = parser.parse_args()
    output = render(args.input)
    print(f'Report: {output}', flush=True)
    if args.open:
        import webbrowser
        if not webbrowser.open(output.as_uri()):
            print('Open report.html directly in your browser.')


if __name__ == '__main__':
    main()
