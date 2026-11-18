import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { LAGS, MODELS } from '../assets/core.js';
import {
  buildScenarioArchive, closestChoice, comparisonPool, findScenarioMatches,
  originOnOrBefore, readScenarioQuery, scenarioCsv, writeScenarioQuery
} from '../assets/scenario-data.js';

const file = name => readFile(new URL('../' + name, import.meta.url), 'utf8');
const json = async name => JSON.parse(await file(name));
const forecasts = (series, horizon, scheme) => json(`data/forecasts/${series}-h${horizon}-${scheme}.json`);
const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 3e-14, `${actual} != ${expected}`);

test('all four lag values and dates agree with source observations at all 18,480 matched origins', async () => {
  const manifest = await json('data/manifest.json');
  let origins = 0;
  for (const dataset of manifest.series) {
    const data = await json(`data/series/${dataset.id}.json`);
    const lines = (await file(dataset.file)).trim().split(/\r?\n/);
    const columns = lines.shift().split(',');
    const dateColumn = columns.indexOf(dataset.financial ? 'Date' : 'date');
    const priceColumn = columns.indexOf(dataset.financial ? 'Adj Close' : 'value');
    const sourceRows = lines.map(line => line.split(','));
    const dates = new Map(sourceRows.map((row, i) => [row[dateColumn], i]));
    for (const horizon of [1,5]) for (const scheme of ['rolling','expanding']) {
      const d = await forecasts(dataset.id, horizon, scheme);
      const archive = buildScenarioArchive(d, data);
      assert.equal(archive.length, 924);
      for (const row of archive) {
        const index = dates.get(row.date);
        for (const [j, lag] of LAGS.entries()) {
          const earlier = sourceRows[index - lag];
          const value = Number(earlier[priceColumn]);
          const expected = dataset.financial ? Math.log(value / Number(sourceRows[index - lag - 1][priceColumn])) : value;
          near(row.values[j], expected);
          assert.equal(row.lagDates[j], earlier[dateColumn]);
        }
      }
      origins += archive.length;
    }
  }
  assert.equal(origins, 18480);
});

test('earlier outcomes enforce the target date boundary over five-day targets, weekends and fold gaps', async () => {
  const data = await forecasts('dax',5,'expanding');
  const rows = buildScenarioArchive(data, await json('data/series/dax.json'));
  const ref = rows[231];
  assert.equal(ref.date, '1994-02-23');
  assert.equal(ref.targetDate, '1994-03-02');
  assert.equal(data.y_true[ref.index],-0.0584738874811768);
  assert.equal(data.predictions.Linear[ref.index],0.003260521945078);
  assert.equal(data.predictions.RandomForest[ref.index],-0.0018904065911251);
  [0.002643407773195798,0.002128496665756488,-0.007865564230974944,-0.0031228694937855215].forEach((item,i) => near(ref.values[i], item));
  const sample = rows[15];
  const pool = comparisonPool(rows,sample,'past');
  assert.ok(pool.candidates.length > 0);
  assert.ok(pool.candidates.every(row => row.targetDate < sample.date));
  assert.ok(rows.some(row => row.targetDate === sample.date));
  assert.equal(pool.candidates.length, rows.filter(row => row.targetDate < sample.date).length);
  assert.equal(comparisonPool(rows,rows[0],'past').candidates.length,0);
  assert.deepEqual(findScenarioMatches([],rows[0].values,5,0).matches,[]);
  assert.equal(originOnOrBefore(rows,'1994-02-22'),230);
  assert.equal(originOnOrBefore(rows,ref.date),231);
});

test('a zero Sunspot profile preserves all 152 exact matches, the reference date, and full-precision records',async () => {
  const d = await forecasts('monash_sunspot',1,'rolling');
  const archive = buildScenarioArchive(d, await json('data/series/monash_sunspot.json'));
  assert.deepEqual(archive[0].values,[11,88,69,0]);
  const ref = archive[923];
  assert.equal(ref.date,'2020-05-26');
  const settings = {profile:[0,0,0,0],count:5,pool:'all',selectedDate:archive[72].date};
  const pool = comparisonPool(archive,ref,settings.pool);
  const result = findScenarioMatches(pool.candidates,settings.profile,settings.count,ref.index);
  assert.equal(result.exactCount,152);
  assert.equal(result.matches.length,152);
  assert.equal(result.matches[0].row.date,ref.date);
  assert.equal(result.matches[1].row.date,'2017-11-24');
  assert.ok(result.matches.every(match => match.distance === 0));
  const observed = result.matches.map(({row})=>d.y_true[row.index]);
  assert.deepEqual([Math.min(...observed), Math.max(...observed)],[0,17]);
  const [header,...csv] = scenarioCsv(d,ref,settings,result.matches,pool.candidates.length);
  assert.equal(csv.length,152 * MODELS.length);
  const records = csv.map(values=>Object.fromEntries(header.map((name,i)=>[name,values[i]])));
  for (const [i, match] of result.matches.entries()) {
    const index = match.row.index;
    for (const [j,model] of MODELS.entries()) {
      const record = records[i*7+j];
      assert.equal(record.model,model);
      assert.equal(record.origin_date,match.row.date);
      assert.equal(record.input_rms_distance,0);
      assert.equal(record.model_native_fold,d.folds[model][index]);
      assert.equal(record.y_pred,d.predictions[model][index]);
      assert.equal(record.y_true,d.y_true[index]);
      assert.equal(record.displayed_date,match.row.date === settings.selectedDate);
    }
  }
});

test('distance uses the four inputs and retains a tie at a nonzero cutoff',() => {
  const rows = [0,2,-2,4,6].map((n,index)=>({index, values:[n,0,0,0]}));
  const result = findScenarioMatches(rows,[0,0,0,0],2,0);
  assert.deepEqual(result.matches.map(match=>[match.row.index,match.distance]),[[0,0],[1,1],[2,1]]);
  assert.equal(closestChoice([-2,0,2],1),1);
  assert.equal(closestChoice([-2,0,2],99),2);
  assert.equal(closestChoice([5],-200),0);
});

test('share state round-trips numerical precision, zero values, visible date, and search period',() => {
  const settings = {pool:'all',count:25,profile:[-0.0185591053980085,0,1e-8,0.5],selectedDate:'2022-01-19'};
  const params = new URLSearchParams();
  writeScenarioQuery(params,settings);
  assert.deepEqual(readScenarioQuery(new URLSearchParams(params.toString())),settings);
  for (const invalid of ['0,,0,0','a,0,0,0','0,1,2','1,2,3,Infinity']) {
    assert.deepEqual(readScenarioQuery(new URLSearchParams({pool:'bad', k:'0', lags:invalid, case:'broken'})),
      {pool:'past',count:5,profile:null,selectedDate:null});
  }
});
