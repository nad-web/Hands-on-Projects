import { MODELS, MODEL, LAGS, esc, fmt, pct, dateLabel } from './core.js';
import {
  MATCH_COUNTS, FORECAST_TABLE, buildScenarioArchive, closestChoice, comparisonPool,
  findScenarioMatches, originOnOrBefore, scenarioCsv
} from './scenario-data.js';

export function mountScenarioExplorer(root, options) {
  const { forecasts, observations, notebooks, observationFile, url, onChange, onReplay, onExport } = options;
  const archive = buildScenarioArchive(forecasts, observations);
  const isFinancial = forecasts.series !== 'monash_sunspot';
  const unit = value => isFinancial ? pct(value, 3) : fmt(value, 2);
  const distanceUnit = value => isFinancial ? `${fmt(value * 100, 4)} pp` : `${fmt(value, 3)} units`;
  const settings = { ...options.settings };
  let reference = archive[options.anchorIndex];
  let activeModel = options.model;
  let pool;
  let matches = [];
  let exactCount = 0;
  let selected;

  const find = selector => root.querySelector(selector);
  const columns = LAGS.map((lag, feature) => {
    const steps = isFinancial ? 'sessions' : 'days';
    const label = lag === 1 ? (isFinancial ? '1 session earlier' : '1 day earlier') : `${lag} ${steps} earlier`;
    return `<label class="sc-feature" for="sc-lag-${lag}">
      <span class="sc-feature-label"><span>${label}</span><output id="sc-value-${lag}" for="sc-lag-${lag}"></output></span>
      <input type="range" id="sc-lag-${lag}" data-scenario-feature="${feature}" min="0" step="1" aria-label="${label}">
      <span class="sc-feature-bounds"><span id="sc-min-${lag}"></span><span id="sc-max-${lag}"></span></span>
    </label>`;
  }).join('');
  const modelRows = MODELS.map(model => `<button class="sc-prediction" type="button" data-scenario-model="${model}" aria-pressed="false" style="--model-color:${MODEL[model].color}">
    <span class="sc-model-name">${esc(MODEL[model].name)}</span>
    <span class="sc-track" aria-hidden="true"><i class="sc-observed-tick"></i><i class="sc-model-dot"></i></span>
    <strong class="sc-point"></strong>
  </button>`).join('');

  root.innerHTML = `<div class="sc-layout">
    <section class="panel sc-controls" aria-labelledby="sc-input-heading">
      <h2 id="sc-input-heading">Lagged observations</h2>
      <label class="sc-field" for="sc-anchor">Reference forecast date</label>
      <div class="sc-date-control">
        <button class="btn small" type="button" data-sc-action="previous" aria-label="Previous reference origin">‹</button>
        <input id="sc-anchor" type="date" min="${archive[0].date}" max="${archive.at(-1).date}" value="${reference.date}">
        <button class="btn small" type="button" data-sc-action="next" aria-label="Next reference origin">›</button>
      </div>
      <label class="sc-field" for="sc-pool">Search dates</label>
      <select id="sc-pool"><option value="past">Earlier observed outcomes</option><option value="all">Full archive</option></select>
      <p id="sc-eligible" class="sc-hint"></p>
      <div class="sc-controls-intro"><h3>Choose a profile</h3><button type="button" class="quiet" data-sc-action="reset">Use reference values</button></div>
      <p class="sc-hint">Each slider steps through observed ${isFinancial ? 'one-session log returns' : 'sunspot values'}, ordered from low to high.</p>
      <div class="sc-features">${columns}</div>
      <div class="sc-count"><label for="sc-count">Closest dates to keep</label>
        <select id="sc-count">${MATCH_COUNTS.map(n => `<option value="${n}">${n}</option>`).join('')}</select>
      </div>
      <p class="sc-hint">Dates tied at the cutoff are included.</p>
      <button type="button" class="sc-quick-result" data-sc-action="view-results" id="sc-quick-result"></button>
      <p id="sc-notice" class="sc-hint sc-notice" role="status"></p>
    </section>
    <section class="panel sc-results" aria-labelledby="sc-results-heading">
      <div class="sc-results-head"><h2 id="sc-results-heading">Recorded forecasts</h2><span id="sc-match-count" class="sc-count-label" aria-live="polite"></span></div>
      <div id="sc-empty" class="sc-empty" hidden>
        <h3>No earlier outcomes in this setting</h3>
        <p>Choose a later reference origin or search the full archive to compare recorded forecasts.</p>
        <button class="btn" type="button" data-sc-action="all">Search the full archive</button>
      </div>
      <div id="sc-match-content">
        <label class="sc-field" for="sc-match">Inspect a matching date</label>
        <select id="sc-match"></select>
        <p id="sc-match-description" class="sc-hint"></p>
        <div class="sc-stat-pair">
          <div><span>Observed target</span><strong id="sc-actual"></strong><small id="sc-target-date"></small></div>
          <div><span>Input distance</span><strong id="sc-distance"></strong><small>Root mean square gap across four lags</small></div>
        </div>
        <div class="sc-chart-heading"><h3>Model forecasts</h3><div class="sc-legend"><i class="sc-legend-dot"></i> Forecast <i class="sc-legend-line"></i> Observed</div></div>
        <p class="sc-hint" id="sc-forecast-units"></p>
        <div id="sc-forecasts" role="group" aria-label="Inspect a model’s recorded forecast">${modelRows}
          <div class="sc-prediction sc-benchmark" style="--model-color:var(--faint)"><span class="sc-model-name">Persistence</span><span class="sc-track" aria-hidden="true"><i class="sc-observed-tick"></i><i class="sc-model-dot"></i></span><strong class="sc-point"></strong></div>
        </div>
        <div class="sc-axis" aria-hidden="true"><span></span><div><span id="sc-axis-min"></span><span id="sc-axis-max"></span></div><span></span></div>
        <div class="sc-model-detail" aria-live="polite">
          <div class="sc-model-heading"><h3 id="sc-model-title"></h3><a id="sc-model-notebook">Read notebook ↗</a></div>
          <dl><div><dt>Absolute error on this date</dt><dd id="sc-error"></dd></div><div><dt id="sc-mae-label">Mean absolute error</dt><dd id="sc-mae"></dd></div></dl>
          <p class="sc-hint" id="sc-folds"></p>
        </div>
        <div class="sc-actions"><button class="btn" type="button" data-sc-action="replay">Replay this date →</button><button class="btn" type="button" data-sc-action="export">Export matches as CSV</button></div>
      </div>
    </section>
  </div>
  <details class="panel sc-inspect" id="sc-lag-detail"><summary>Inspect the four lag values and their source dates</summary>
    <div id="sc-input-comparison"></div>
    <div class="sc-source"><a href="${esc(url(observationFile))}" download>Download source observations</a><a href="${esc(url(FORECAST_TABLE))}" download>Download matched forecast table</a></div>
  </details>
  <details class="panel sc-inspect" id="sc-method"><summary>How historical matching works</summary>
    <p>The reference date starts the sliders at an observed profile. Each candidate receives the square root of the mean of its four squared lag differences. All four lags have equal weight. ${isFinancial ? 'A gap of 0.10 percentage points corresponds to 0.001 in stored log-return units.' : 'Distances use the same native units as the sunspot observations.'}</p>
    <p>The earlier-outcomes option includes dates whose target was observed strictly before the reference date. The full archive includes all ${archive.length} dates in this setting. Matching uses lag values only. Equal distances use chronological order, with an exact reference-date match first when eligible.</p>
    <p>Linear and the tree models use these four tabular inputs. ARIMA uses the observed history through the origin. LSTM and the simplified TFT use the 21 preceding feature rows. The predictions come from the fitted models at the matching date. The mean error describes the dates selected here.</p>
    <p>Financial CSV values are stored as decimal log returns. The displayed percentages multiply those values by 100. The export includes the chosen inputs, observed inputs, source dates, native folds and full-precision forecasts.</p>
    <a href="${esc(url('documentation/methods.html'))}#historical-scenarios">Read the method and the worked example ↗</a>
  </details>
  <div class="sc-examples">
    <aside class="sc-example"><div><h3>A five-session DAX decline</h3><p>On 23 February 1994, Linear forecast a 0.326% cumulative log return. The observed return was −5.847%. Compare all seven forecasts for that date.</p></div><button type="button" class="btn" data-sc-action="dax">Open DAX example →</button></aside>
    <aside class="sc-example"><div><h3>Same profile, different outcomes</h3><p>Four zero lags occur on 152 dates in the rolling one-day Sunspot experiment. Inspect the forecasts for two of those dates.</p></div><button type="button" class="btn" data-sc-action="sunspot">Open Sunspot example →</button></aside>
  </div>`;

  const resultNodes = new Map(MODELS.map(model => [model, find(`[data-scenario-model="${model}"]`)]));
  const countSelect = find('#sc-count');
  const poolSelect = find('#sc-pool');
  const matchSelect = find('#sc-match');
  const dateInput = find('#sc-anchor');
  countSelect.value = String(settings.count);
  poolSelect.value = settings.pool;

  function persist() {
    onChange(reference.index, activeModel, { ...settings, profile: settings.profile.slice() });
  }

  function notice(message = '') { find('#sc-notice').textContent = message; }

  function configureSliders(requestedProfile) {
    pool = comparisonPool(archive, reference, settings.pool);
    settings.profile = LAGS.map((lag, feature) => {
      const values = pool.choices[feature];
      const target = requestedProfile?.[feature] ?? reference.values[feature];
      const choice = closestChoice(values, target);
      const slider = find(`#sc-lag-${lag}`);
      slider.max = String(values.length - 1);
      slider.value = String(choice);
      slider.disabled = values.length < 2;
      find(`#sc-min-${lag}`).textContent = unit(values[0]);
      find(`#sc-max-${lag}`).textContent = unit(values.at(-1));
      return values[choice];
    });
    find('[data-sc-action="previous"]').disabled = reference.index === 0;
    find('[data-sc-action="next"]').disabled = reference.index === archive.length - 1;
    find('#sc-eligible').textContent = settings.pool === 'past'
      ? `${pool.candidates.length} of ${archive.length} dates have a target observed before ${dateLabel(reference.date)}.`
      : `All ${archive.length} matched forecast dates in this dataset, horizon and window.`;
  }

  function updateProfileValues() {
    LAGS.forEach((lag, feature) => {
      const value = unit(settings.profile[feature]);
      find(`#sc-value-${lag}`).textContent = value;
      find(`#sc-lag-${lag}`).setAttribute('aria-valuetext', value);
    });
  }

  function showInputComparison() {
    if (!selected) {
      find('#sc-input-comparison').textContent = 'Select a matching date to compare its observed lags with the slider profile.';
      return;
    }
    const body = LAGS.map((lag, feature) => `<tr><th scope="row">lag_${lag}</th>
      <td>${unit(settings.profile[feature])}</td><td>${unit(selected.row.values[feature])}</td><td>${dateLabel(selected.row.lagDates[feature])}</td></tr>`).join('');
    find('#sc-input-comparison').innerHTML = `<div class="table-wrap"><table><caption>Inputs at ${dateLabel(selected.row.date)}</caption><thead><tr><th>Lag</th><th>Slider profile</th><th>Observed input</th><th>Observation date</th></tr></thead><tbody>${body}</tbody></table></div>`;
  }

  function updateFocusedModel() {
    if (!selected) return;
    const index = selected.row.index;
    const predictions = forecasts.predictions[activeModel];
    const error = Math.abs(predictions[index] - forecasts.y_true[index]);
    const totalError = matches.reduce((sum, { row }) => sum + Math.abs(predictions[row.index] - forecasts.y_true[row.index]), 0);
    for (const [model, button] of resultNodes) button.setAttribute('aria-pressed', String(model === activeModel));
    find('#sc-quick-result').textContent = `${dateLabel(selected.row.date)} · ${MODEL[activeModel].name} ${unit(predictions[index])} · Observed ${unit(forecasts.y_true[index])} ↓`;
    find('#sc-model-title').textContent = MODEL[activeModel].name;
    find('#sc-error').textContent = unit(error);
    find('#sc-mae').textContent = unit(totalError / matches.length);
    find('#sc-mae-label').textContent = `Mean absolute error across ${matches.length} ${matches.length === 1 ? 'date' : 'dates'}`;
    find('#sc-folds').textContent = `Matched block ${forecasts.matched_fold[index] + 1} · ${MODEL[activeModel].name} fit ${forecasts.folds[activeModel][index] + 1}. The CSV uses zero-based fold numbers.`;
    const notebook = notebooks.find(item => item.id === activeModel);
    find('#sc-model-notebook').href = url(`documentation/notebooks/${notebook.slug}.html`);
    find('[data-sc-action="replay"]').setAttribute('aria-label', `Replay ${MODEL[activeModel].name} on ${selected.row.date}`);
  }

  function displayMatch() {
    find('#sc-quick-result').hidden = !selected;
    find('#sc-empty').hidden = Boolean(selected);
    find('#sc-match-content').hidden = !selected;
    if (!selected) { showInputComparison(); return; }
    const { row, distance } = selected;
    const index = row.index;
    const target = forecasts.y_true[index];
    const baseline = forecasts.random_walk_pred[index];
    const points = MODELS.map(model => forecasts.predictions[model][index]);
    const min = Math.min(target, baseline, ...points);
    const max = Math.max(target, baseline, ...points);
    const padding = (max - min || Math.max(Math.abs(min) * .1, isFinancial ? .001 : 1)) * .10;
    const left = min - padding;
    const right = max + padding;
    const position = value => `${100 * (value - left) / (right - left)}%`;
    const chart = find('#sc-forecasts');
    chart.style.setProperty('--realized', position(target));
    for (const [model, button] of resultNodes) {
      const value = forecasts.predictions[model][index];
      button.style.setProperty('--forecast', position(value));
      button.querySelector('.sc-point').textContent = unit(value);
      button.setAttribute('aria-label', `Inspect ${MODEL[model].name}, forecast ${unit(value)}`);
    }
    const persistence = find('.sc-benchmark');
    persistence.style.setProperty('--forecast', position(baseline));
    persistence.querySelector('.sc-point').textContent = unit(baseline);
    find('#sc-axis-min').textContent = unit(left);
    find('#sc-axis-max').textContent = unit(right);
    find('#sc-actual').textContent = unit(target);
    find('#sc-target-date').textContent = `Observed ${dateLabel(row.targetDate)}`;
    find('#sc-distance').textContent = distance === 0 ? 'Exact match' : distanceUnit(distance);
    find('#sc-forecast-units').textContent = `${isFinancial ? 'Cumulative future log return' : 'Future sunspot value'} · forecast issued ${dateLabel(row.date)}. Select a model to inspect its error.`;
    matchSelect.value = String(index);
    settings.selectedDate = row.date;
    showInputComparison();
    updateFocusedModel();
  }

  function search(requestedDate = null) {
    updateProfileValues();
    ({ matches, exactCount } = findScenarioMatches(pool.candidates, settings.profile, settings.count, reference.index));
    selected = matches.find(match => match.row.date === requestedDate) || matches[0];
    find('#sc-match-count').textContent = `${matches.length} ${matches.length === 1 ? 'match' : 'matches'}`;
    if (!selected) settings.selectedDate = null;
    matchSelect.innerHTML = matches.map(({ row, distance }, rank) =>
      `<option value="${row.index}">${rank + 1}. ${dateLabel(row.date)} · ${distance === 0 ? 'exact' : distanceUnit(distance)}</option>`).join('');
    const extra = matches.length > settings.count ? ' All ties at the cutoff are shown.' : '';
    const targetRange = matches.map(match => forecasts.y_true[match.row.index]);
    let description = `${matches.length} of ${pool.candidates.length} eligible dates selected by input distance.${extra}`;
    if (exactCount) description += ` ${exactCount} ${exactCount === 1 ? 'date matches' : 'dates match'} all four inputs exactly.`;
    if (matches.length > 1) description += ` Observed target range ${unit(Math.min(...targetRange))} to ${unit(Math.max(...targetRange))}.`;
    find('#sc-match-description').textContent = description;
    displayMatch();
    persist();
  }

  function changeReference(index, calendarDate) {
    if (!archive[index]) return;
    reference = archive[index];
    dateInput.value = reference.date;
    notice(calendarDate && calendarDate !== reference.date
      ? `The closest available origin on or before ${calendarDate} is ${reference.date}.`
      : 'Sliders now show the observed profile at the reference date.');
    configureSliders(reference.values);
    search();
  }

  function changePool(scope) {
    const previous = settings.profile;
    settings.pool = scope;
    poolSelect.value = scope;
    configureSliders(previous);
    const snapped = previous.some((value, feature) => value !== settings.profile[feature]);
    notice(snapped ? 'The sliders moved to the closest values available in this search period.' : '');
    search(settings.selectedDate);
  }

  function exportMatches() {
    if (!selected) return false;
    const rows = scenarioCsv(forecasts, reference, settings, matches, pool.candidates.length);
    const name = `historical-scenarios-${forecasts.series}-h${forecasts.horizon}-${forecasts.scheme}-${reference.date}.csv`;
    onExport(name, rows);
    return true;
  }

  root.addEventListener('input', event => {
    const featureId = event.target.dataset.scenarioFeature;
    if (featureId === undefined) return;
    const feature = Number(featureId);
    settings.profile[feature] = pool.choices[feature][Number(event.target.value)];
    notice();
    search();
  });
  root.addEventListener('change', event => {
    const element = event.target;
    if (element === poolSelect) changePool(element.value);
    if (element === countSelect) {
      settings.count = Number(element.value);
      search(settings.selectedDate);
    }
    if (element === matchSelect) {
      selected = matches.find(match => match.row.index === Number(element.value));
      displayMatch();
      persist();
    }
    if (element === dateInput) {
      if (!element.value || !element.validity.valid) {
        element.value = reference.date;
        notice(`Choose a date from ${archive[0].date} to ${archive.at(-1).date}.`);
        return;
      }
      changeReference(originOnOrBefore(archive, element.value), element.value);
    }
  });
  root.addEventListener('click', event => {
    const button = event.target.closest('[data-scenario-model], [data-sc-action]');
    if (!button) return;
    const model = button.dataset.scenarioModel;
    if (model) { activeModel = model; updateFocusedModel(); persist(); return; }
    switch (button.dataset.scAction) {
      case 'previous': changeReference(reference.index - 1); break;
      case 'next': changeReference(reference.index + 1); break;
      case 'all': changePool('all'); break;
      case 'reset': configureSliders(reference.values); notice('Reference values restored.'); search(); break;
      case 'replay': if (selected) onReplay(selected.row.index, activeModel); break;
      case 'export': exportMatches(); break;
      case 'dax':
      case 'sunspot': options.onExample(button.dataset.scAction); break;
      case 'view-results': find('#sc-results-heading').scrollIntoView({block:'start'}); break;
    }
  });
  configureSliders(settings.profile || reference.values);
  search(settings.selectedDate);
  return { exportMatches };
}
