import { LAGS, MODELS } from './core.js';

export const MATCH_COUNTS = [5, 10, 25];
export const FORECAST_TABLE = 'research/results/cross_model/tables/matched_observation_predictions.csv';

export function readScenarioQuery(params) {
  const parts = params.get('lags')?.split(',');
  const validProfile = parts?.length === LAGS.length &&
    parts.every(part => part.trim() !== '' && Number.isFinite(Number(part)));
  const requestedCount = Number(params.get('k'));
  const selectedDate = params.get('case');
  return {
    pool: params.get('pool') === 'all' ? 'all' : 'past',
    count: MATCH_COUNTS.includes(requestedCount) ? requestedCount : 5,
    profile: validProfile ? parts.map(Number) : null,
    selectedDate: /^\d{4}-\d{2}-\d{2}$/.test(selectedDate || '') ? selectedDate : null
  };
}

export function writeScenarioQuery(params, settings) {
  params.set('pool', settings.pool);
  params.set('k', String(settings.count));
  if (settings.profile) params.set('lags', settings.profile.join(','));
  if (settings.selectedDate) params.set('case', settings.selectedDate);
}

// Align on the original ISO date. A financial signal is a one-observation log
// return ending on that date. The Sunspot signal is the observed native value.
export function buildScenarioArchive(forecasts, observations) {
  const positions = new Map(observations.dates.map((date, index) => [date, index]));
  return forecasts.origin_date.map((date, index) => {
    const position = positions.get(date);
    const values = LAGS.map(lag => observations.signal[position - lag]);
    if (!values.every(Number.isFinite)) throw new Error(`Missing lag history for ${date}.`);
    return {
      index, date, targetDate: forecasts.target_date[index], values,
      lagDates: LAGS.map(lag => observations.dates[position - lag])
    };
  });
}

export function originOnOrBefore(rows, date) {
  // The source's matched observations are chronological. Calendar gaps can
  // include weekends and the warm-up periods between held-out blocks.
  let low = 0;
  let high = rows.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (rows[middle].date <= date) low = middle + 1;
    else high = middle;
  }
  return Math.max(0, low - 1);
}

export function comparisonPool(archive, reference, scope) {
  const candidates = scope === 'all'
    ? archive
    : archive.filter(row => row.targetDate < reference.date);
  const choices = LAGS.map((_, feature) => {
    const observed = new Set(candidates.map(row => row.values[feature]));
    observed.add(reference.values[feature]);
    return [...observed].sort((a, b) => a - b);
  });
  return { candidates, choices };
}

export function closestChoice(sorted, value) {
  let low = 0;
  let high = sorted.length - 1;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (sorted[middle] < value) low = middle + 1;
    else high = middle;
  }
  if (low > 0 && Math.abs(sorted[low - 1] - value) <= Math.abs(sorted[low] - value)) return low - 1;
  return low;
}

// Only the four lagged observations enter the distance. Model predictions,
// realised outcomes, and losses are never used as similarity criteria.
export function findScenarioMatches(candidates, profile, count, referenceIndex) {
  const ranked = candidates.map(row => {
    let squaredGap = 0;
    for (let feature = 0; feature < LAGS.length; feature++) {
      squaredGap += (row.values[feature] - profile[feature]) ** 2;
    }
    return { row, distance: Math.sqrt(squaredGap / LAGS.length) };
  });
  ranked.sort((left, right) => {
    const difference = left.distance - right.distance;
    if (difference !== 0) return difference;
    if (left.distance === 0) {
      if (left.row.index === referenceIndex && right.row.index !== referenceIndex) return -1;
      if (right.row.index === referenceIndex && left.row.index !== referenceIndex) return 1;
    }
    return left.row.index - right.row.index;
  });
  if (!ranked.length) return { matches: [], exactCount: 0 };
  const threshold = ranked[Math.min(count, ranked.length) - 1].distance;
  return {
    matches: ranked.filter(match => match.distance <= threshold),
    exactCount: ranked.filter(match => match.distance === 0).length
  };
}

export function scenarioCsv(forecasts, reference, settings, matches, eligibleCount) {
  const header = [
    'series', 'horizon', 'scheme', 'reference_origin', 'pool', 'requested_matches',
    'eligible_origins', 'returned_matches_including_ties', ...LAGS.map(lag => `query_lag_${lag}`),
    'match_rank', 'input_rms_distance', 'origin_date', 'target_date', 'origin_index',
    'matched_fold', ...LAGS.map(lag => `observed_lag_${lag}`), ...LAGS.map(lag => `lag_${lag}_date`),
    'model', 'model_native_fold', 'y_pred', 'y_true', 'persistence', 'absolute_error', 'displayed_date'
  ];
  const rows = [header];
  matches.forEach(({ row, distance }, matchIndex) => {
    const index = row.index;
    for (const model of MODELS) {
      const prediction = forecasts.predictions[model][index];
      const target = forecasts.y_true[index];
      rows.push([
        forecasts.series, forecasts.horizon, forecasts.scheme, reference.date, settings.pool,
        settings.count, eligibleCount, matches.length, ...settings.profile, matchIndex + 1,
        distance, row.date, row.targetDate, forecasts.origin_index[index], forecasts.matched_fold[index],
        ...row.values, ...row.lagDates, model, forecasts.folds[model][index], prediction, target,
        forecasts.random_walk_pred[index], Math.abs(prediction - target), row.date === settings.selectedDate
      ]);
    }
  });
  return rows;
}
